# Base VAE class definition
import numpy as np
import torch, cv2
import torch.nn as nn
import torch.distributions as dist
from models import encoders, decoders
from utils import get_mean, kl_divergence, Constants, create_vocab, W2V, load_images, lengths_to_mask
from vis import t_sne, tensors_to_df, plot_embeddings, plot_kls_df
from torch.utils.data import DataLoader
import pickle, os
from data_proc.process_audio import numpy_to_wav
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, enc, dec, data_path, feature_dim, n_latents, prior_dist=dist.Normal, likelihood_dist=dist.Normal, post_dist=dist.Normal):
        super(VAE, self).__init__()
        self.device = None
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0
        self.pth = data_path
        self.data_dim = feature_dim
        self.n_latents = n_latents
        self.enc_name, self.dec_name = enc, dec
        self.enc, self.dec = self.get_nework_classes(enc, dec)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False)  # logvar
        ])
        self.modelName = 'vae_{}'.format(os.path.basename(data_path))
        self.w2v = W2V(feature_dim, self.pth) if "word2vec" in self.pth else None

    def get_nework_classes(self, enc, dec):
       assert hasattr(encoders, "Enc_{}".format(enc)), "Did not find encoder {}".format(enc)
       enc_obj = getattr(encoders, "Enc_{}".format(enc))(self.n_latents, self.data_dim)
       assert hasattr(decoders, "Dec_{}".format(dec)), "Did not find decoder {}".format(enc)
       dec_obj = getattr(decoders, "Dec_{}".format(dec))(self.n_latents, self.data_dim)
       return enc_obj, dec_obj

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    def getDataLoaders(self, batch_size,device="cuda"):
        self.device = device
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        if not ".pkl" in self.pth:
            if "image" in self.pth:
                d = load_images(self.pth, self.data_dim)
            else: raise Exception("If {} is an image dataset, please include 'image' in it's name. "
                                  "For other data types you should use .pkl or write your own dataLoader'".format(self.pth))
        else:
            with open(self.pth, 'rb') as handle:
                d = pickle.load(handle)
            if "word2vec" in self.pth:
                d = d.reshape(d.shape[0],-1)
                #d = self.w2v.normalize_w2v(d)
                if len(d.shape) < 2: d = np.expand_dims(d, axis=1)
            elif self.enc.name in ["Transformer", "CNN"]:
                d = [torch.from_numpy(np.asarray(x).astype(np.float)) for x in d] if self.enc_name.lower() == "transformerimg" else [torch.from_numpy(np.asarray(x[0])) for x in d]
                if self.enc_name.lower() == "cnn":
                    d = torch.stack(d).transpose(1,3) #.reshape(len(d), -1)
                else:
                    if len(d[0].shape) < 3:
                        d = [torch.unsqueeze(i, dim=1) for i in d]
                    kwargs["collate_fn"] = self.seq_collate_fn
            elif self.enc.name == "AudioConv":
                d = [torch.from_numpy(np.asarray(x).astype(np.int16)) for x in d]
                d = torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0)
        t_dataset = d[:int(len(d)*(0.9))]
        v_dataset = d[int(len(d)*(0.9)):]
        if self.enc.name != "Transformer":
            t_dataset = torch.utils.data.TensorDataset(torch.tensor(t_dataset))
            v_dataset = torch.utils.data.TensorDataset(torch.tensor(v_dataset))
        train = DataLoader(t_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test = DataLoader(v_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

    def forward(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        if self.dec.name.lower() == "transformer":
            px_z = self.px_z(*self.dec([zs, x[1]] if x is not None else [zs, None]))
        else: px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def seq_collate_fn(self, batch):
        new_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in batch]))).to(self.device)
        return new_batch, masks

    def generate(self, runPath, epoch):
        N, K = 36, 1
        samples = self.generate_samples(N, K).cpu().squeeze()
        r_l = []
        if "image" in self.pth:
            for r, recons_list in enumerate(samples):
                    recon = recons_list.cpu().reshape(*self.data_dim).unsqueeze(0)
                    r_l = np.asarray(recon) if r_l == [] else np.concatenate((r_l, np.asarray(recon)))
            r_l = np.vstack((np.hstack(r_l[:6]), np.hstack(r_l[6:12]), np.hstack(r_l[12:18]), np.hstack(r_l[18:24]),  np.hstack(r_l[24:30]),  np.hstack(r_l[30:36])))
            cv2.imwrite('{}/visuals/gen_samples_{:03d}.png'.format(runPath, epoch), r_l*255)

    def reconstruct(self, data, runPath, epoch, N=3):
        recons_mat = self.reconstruct_data(data[:N]).squeeze().cpu()
        if ".pkl" in self.pth and self.w2v:
            _data = data[:N].cpu()
            target, reconstruct = [], []
            _data = _data.reshape(-1, self.data_dim[-1], self.data_dim[-2])
            recon = recons_mat.reshape(-1, self.data_dim[-1], self.data_dim[-2])
            for s in _data:
                seq = []
                [seq.append(self.w2v.model.wv.most_similar(positive=[self.w2v.unnormalize_w2v(np.asarray(w)), ])[0][0]) for w in s]
                target.append(" ".join(seq))
            for s in recon:
                seq, prob = [], []
                for w in s:
                    seq.append(self.w2v.model.wv.most_similar(positive=[self.w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0])
                    prob.append("({})".format(str(round(self.w2v.model.wv.most_similar(positive=[self.w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][1], 2))))
                j = [" ".join((x, prob[y])) for y,x in enumerate(seq)]
                reconstruct.append(" ".join(j))
            output = open('{}/visuals/recon_{:03d}.txt'.format(runPath, epoch), "w")
            output.writelines(["|".join(target) + "\n", "|".join(reconstruct)])
            output.close()
        elif self.enc_name.lower() in ["transformerimg", "cnn"]:
            o_l, r_l = [], []
            N = 3 if self.enc_name.lower() == "transformerimg" else 10
            for r, recons_list in enumerate(recons_mat[:N]):
                    _data = data[0][r].cpu()[:N] if self.enc_name.lower() != "cnn" else data[r].cpu().permute(2,1,0)
                    _data = np.hstack(_data) if len(_data.shape) > 3 else _data
                    recon = recons_list.cpu() if self.enc_name.lower() != "cnn" else  recons_list.cpu().permute(2,1,0)
                    recon = np.hstack(recon) if len(recon.shape) > 3 else recon
                    o_l = np.asarray(_data) if o_l == [] else np.concatenate((o_l, np.asarray(_data)), axis=1)
                    r_l = np.asarray(recon) if r_l == [] else np.concatenate((r_l, np.asarray(recon)), axis=1)
            cv2.imwrite('{}/visuals/recon_epoch{}.png'.format(runPath, epoch),np.vstack((o_l, r_l)) * 255)
        elif self.enc_name.lower() == "audio":
             for i in range(3):
                if epoch < 101:
                    numpy_to_wav(os.path.join(runPath,"visuals/",'orig_epoch{}_s{}.wav'.format(epoch, i)),
                         np.asarray(data[i].cpu()).astype(np.int16), 16000)
                numpy_to_wav(os.path.join(runPath,"visuals/",'recon_epoch{}_s{}.wav'.format(epoch, i)),
                                 np.asarray(recons_mat[i].cpu()).astype(np.int16), 16000)


    def analyse(self, data, runPath, epoch, labels=None):
        zsl, kls_df = self.analyse_data(data, K=1, runPath=runPath, epoch=epoch, labels=labels)
        plot_kls_df(kls_df, '{}/visuals/kl_distance_{:03d}.png'.format(runPath, epoch))

    def generate_samples(self, N, K):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            if self.enc_name == "Transformer":
                px_z = self.px_z(*self.dec([latents, None]))
            else:
                px_z = self.px_z(*self.dec(latents))
            #data = px_z.sample(torch.Size([K]))
            data = get_mean(px_z)
        return data

    def reconstruct_data(self, data):
        self.eval()
        if self.enc_name != "Transformer":
            with torch.no_grad():
                qz_x = self.qz_x(*self.enc(data))
                latents = qz_x.rsample()
                px_z = self.px_z(*self.dec(latents.unsqueeze(0)))
                recon = get_mean(px_z)
            return recon

    def analyse_data(self, data, K, runPath, epoch, labels):
        self.eval()
        with torch.no_grad():
            qz_x, _, zs = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, len(data)])).view(-1, pz.batch_shape[-1]),
                   zs.view(-1, zs.size(-1))]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        t_sne([x.detach().cpu() for x in zss[1:]], runPath, epoch, K, labels)
        return torch.cat(zsl, 0).cpu().numpy(), kls_df
