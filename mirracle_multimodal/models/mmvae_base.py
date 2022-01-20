# Base MMVAE class definition, common for PoE and MoE
from itertools import combinations
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from utils import get_mean, kl_divergence, lengths_to_mask
from vis import t_sne, tensors_to_df, plot_embeddings, plot_kls_df
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import numpy as np
import cv2
from .vae import VAE

class MMVAE(nn.Module):
    def __init__(self, prior_dist, encoders, decoders, data_paths, feature_dims, n_latents):
        super(MMVAE, self).__init__()
        self.device = None
        self.pz = prior_dist
        vae_mods = []
        self.encoders, self.decoders = encoders, decoders
        for e, d, pth, fd in zip(encoders, decoders, data_paths, feature_dims):
            vae_mods.append(VAE(e, d, pth, fd, n_latents))
        self.vaes = nn.ModuleList(vae_mods)
        self.modelName = None  # filled-in per sub-class
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False)  # logvar
        ])
        self.set_likelihood_scaling(feature_dims)

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def set_likelihood_scaling(self, feature_dims):
        max_dimensionality = max([np.prod(x) for x in feature_dims])
        for x in range(len(self.vaes)):
            self.vaes[x].llik_scaling = max_dimensionality / np.prod(feature_dims[x])

    def seq_collate_fn(self, batch):
        new_batch, masks = [], []
        for i, e in enumerate(self.encoders):
            m = [x[i] for x in batch]
            if e == "Transformer":
                masks.append(lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in m]))).to(self.device))
                m = torch.nn.utils.rnn.pad_sequence(m, batch_first=True, padding_value=0.0)
            else:
                masks.append(None)
            new_batch.append(torch.tensor(m).unsqueeze(1))
        new_batch = torch.cat(new_batch, dim=1)
        return new_batch, masks

    def getDataLoaders(self, batch_size, device='cuda'):
        self.device = device
        trains, tests = [], []
        for x in range(len(self.vaes)):
            t, v = self.vaes[x].getDataLoaders(batch_size, device)
            trains.append(t.dataset)
            tests.append(v.dataset)
        train_data = TensorDataset(trains)
        test_data = TensorDataset(tests)
        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        if "Transformer" in self.encoders: kwargs["collate_fn"] = self.seq_collate_fn
        train = DataLoader(train_data, batch_size=batch_size, shuffle=False, **kwargs)
        test = DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

    def generate_samples(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def generate(self, runPath, epoch, N=36):
        for i, samples in enumerate(self.generate_samples(N)):
            samples = samples.reshape(N, *self.vaes[i].data_dim)
            if "image" in self.vaes[i].pth:
                r_l = []
                for r, recons_list in enumerate(samples):
                    recon = recons_list.cpu()
                    recon = recon.reshape(*self.vaes[i].data_dim).unsqueeze(0)
                    r_l = np.asarray(recon) if r_l == [] else np.concatenate((r_l, np.asarray(recon)))
                r_l = np.vstack((np.hstack(r_l[:int(N/6)]), np.hstack(r_l[int(N/6):int(N/3)]), np.hstack(r_l[int(N/3):int(N/2)]), np.hstack(r_l[int(N/2):int(2*N/3)]),
                                 np.hstack(r_l[int(2*N/3):int(5*N/6)]), np.hstack(r_l[int(5*N/6):N])))
                cv2.imwrite('{}/visuals/gen_samples_epoch_{}_m{}.png'.format(runPath, epoch, i), r_l * 255)

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs] if any(isinstance(i, list) for i in px_zs) \
                     else [get_mean(px_z) for px_z in px_zs]
        return recons

    def process_reconstructions(self, recons_mat, data, epoch, runPath, N=8):
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[o][:N].cpu()
                if "word2vec" in self.vaes[o].pth:
                    target, reconstruct = [], []
                    recon = recon.reshape(N, self.vaes[o].data_dim[-1], -1)
                    _data = _data.reshape(N, self.vaes[o].data_dim[-1], -1)
                    for d, rec in zip(_data, recon):
                        seq_d, seq_r = [], []
                        [seq_d.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0]) for w in d]
                        [seq_r.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0]) for w in rec]
                        target.append(" ".join(seq_d))
                        reconstruct.append(" ".join(seq_r))
                    output = open(os.path.join(runPath, "visuals/",'recon_epoch{}_m{}xm{}.txt'.format(epoch, r, o)), "w")
                    output.writelines(["|".join(target)+"\n", "|".join(reconstruct)])
                    output.close()
                elif "image" in self.vaes[o].pth:
                    recon = recon.squeeze(0).cpu()
                    _data = _data.reshape(N, *self.vaes[o].data_dim)
                    recon = recon.reshape(N, *self.vaes[o].data_dim)
                    o_l, r_l = [], []
                    for x in range(_data.shape[0]):
                        org = cv2.copyMakeBorder(np.asarray(_data[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                        rec = cv2.copyMakeBorder(np.asarray(recon[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                        o_l = org if o_l == [] else np.hstack((o_l, org))
                        r_l = rec if r_l == [] else np.hstack((r_l, rec))

                    w2 =cv2.cvtColor(np.vstack((o_l*255, r_l*255)).astype('uint8'), cv2.COLOR_BGR2RGB)
                    w2 = cv2.copyMakeBorder(w2,top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                    cv2.imwrite(os.path.join(runPath,"visuals/",'recon_epoch{}_m{}xm{}.png'.format(epoch, r, o)), w2)

    def analyse(self, data, runPath, epoch):
        zsl, kls_df = self.analyse_data(data, K=10, runPath=runPath, epoch=epoch)
        plot_kls_df(kls_df, '{}/visuals/kl_distance_{}.png'.format(runPath, epoch))

    def analyse_data(self, data, K, runPath, epoch):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss_sampled = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss_sampled)]
            if isinstance(qz_xs, list):
                kls_df = tensors_to_df(
                    [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                     *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                       for p, q in combinations(qz_xs, 2)]],
                    head='KL',
                    keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                          *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                            for i, j in combinations(range(len(qz_xs)), 2)]],
                    ax_names=['Dimensions', r'KL$(q\,||\,p)$'])
            else:
                kls_df = tensors_to_df([kl_divergence(qz_xs, pz).cpu().numpy()], head='KL',
                    keys=[r'KL$(q(z|x)\,||\,p(z))$'], ax_names=['Dimensions', r'KL$(q\,||\,p)$'])
        t_sne(zss_sampled[1:], runPath, epoch)
        return torch.cat(zsl, 0).cpu().numpy(), kls_df
