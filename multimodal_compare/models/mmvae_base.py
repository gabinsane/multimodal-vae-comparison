# Base MMVAE class definition, common for PoE, MoE, MoPoE
from itertools import combinations
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from utils import get_mean, kl_divergence, lengths_to_mask, output_onehot2text
from data_proc.process_audio import numpy_to_wav
from visualization import t_sne, tensors_to_df, plot_embeddings, plot_kls_df
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import numpy as np
from torch.autograd import Variable
import cv2
from .vae import VAE

module_types = {"moe":VAE, "poe":VAE}

class MMVAE(nn.Module):
    def __init__(self, prior_dist, encoders, decoders, data_paths, feature_dims, n_latents, batch_size):
        super(MMVAE, self).__init__()
        self.device = None
        self.pz = prior_dist
        self.batch_size = batch_size
        self.n_latents = n_latents
        vae_mods = []
        self.encoders, self.decoders = encoders, decoders
        for e, d, pth, fd in zip(encoders, decoders, data_paths, feature_dims):
            vae_mods.append(module_types[self.modelName](e, d, pth, fd, n_latents, batch_size))
        self.vaes = nn.ModuleList(vae_mods)
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

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def generate_from_latents(self, latents):
        px_zs = []
        for d, vae in enumerate(self.vaes):
                if "transformer" in self.vaes[d].dec_name.lower():
                    px_zs.append(vae.px_z(*vae.dec([latents, None])))
                else:
                    px_zs.append(vae.px_z(*vae.dec(latents)))
        return px_zs

    def seq_collate_fn(self, batch):
        new_batch, masks = [], []
        for i, e in enumerate(self.encoders):
            m = [x[i] for x in batch]
            if "transformer" in e.lower():
                masks.append(lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in m]))))
                m = list(torch.nn.utils.rnn.pad_sequence(m, batch_first=True, padding_value=0.0))
            else:
                masks.append(None)
            new_batch.append([m])
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
        if any(["transformer" in e.lower() for e in self.encoders]): kwargs["collate_fn"] = self.seq_collate_fn
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
                if vae.dec_name == "Transformer":
                    px_z = vae.px_z(*vae.dec([latents, None]))
                else:
                    px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def generate(self, runPath, epoch, N=36):
        for i, samples in enumerate(self.generate_samples(N)):
            if "image" in self.vaes[i].pth:
                samples = samples.reshape(N, *self.vaes[i].data_dim)
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
            if self.modelName != "mopoe":
                _, px_zs, _ = self.forward(data)
            else:
                _, px_zs, _, _ = self.forward(data)
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs] if any(isinstance(i, list) for i in px_zs) \
                     else [get_mean(px_z) for px_z in px_zs]
        return recons

    def process_reconstructions(self, recons_mat, data, epoch, runPath, N=8):
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                if self.vaes[o].enc.name == "CNN":
                    _data = torch.stack(data[o]).cpu() if isinstance(data[0], list) else data[o].cpu()
                    recon = recon.squeeze(0).cpu()
                    if "cub_" in self.vaes[o].pth:
                        _data = _data.permute(0,3,2,1)
                    elif self.vaes[o].enc_name in ["MNIST", "SVHN"]:
                        if self.vaes[o].enc_name == "MNIST":
                            _data = _data.permute(0,2,3,1).cpu()
                            recon = recon.unsqueeze(-1).cpu()
                        else:
                            _data = _data.reshape(-1, 32,32,3)
                    else:
                        _data = _data.reshape(len(_data), *self.vaes[o].data_dim)
                    o_l, r_l = [], []
                    for x in range(recon.shape[0]):
                        org = cv2.copyMakeBorder(np.asarray(_data[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[211,211,211])
                        rec = cv2.copyMakeBorder(np.asarray(recon[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[211,211,211])
                        o_l = org if o_l == [] else np.hstack((o_l, org))
                        r_l = rec if r_l == [] else np.hstack((r_l, rec))
                    w2 =cv2.cvtColor(np.vstack((o_l*255, r_l*255)).astype('uint8'), cv2.COLOR_BGR2RGB)
                    w2 = cv2.copyMakeBorder(w2,top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=[211,211,211])
                    cv2.imwrite(os.path.join(runPath,"visuals/",'recon_epoch{}_m{}xm{}.png'.format(epoch, r, o)), w2)
                elif self.vaes[o].enc_name.lower() == "transformerimg":
                    _data = torch.stack(data[o][0]).cpu()[:N]
                    o_l, r_l = [], []
                    for re, recons_list in enumerate(recon[:N]):
                        d = _data[re].cpu().permute(0, 1, 2, 3)
                        recon = recons_list.cpu()
                        o_l = np.asarray(np.hstack(d)) if o_l == [] else np.concatenate(
                            (o_l, np.asarray(np.hstack(d))), axis=1)
                        r_l = np.asarray(np.hstack(recon)) if r_l == [] else np.concatenate(
                            (r_l, np.asarray(np.hstack(recon))), axis=1)
                    cv2.imwrite(os.path.join(runPath,"visuals/",'recon_epoch{}_minp{}.png'.format(epoch, r)),
                                np.vstack((o_l, r_l)) * 255)
                elif self.vaes[o].enc_name.lower() in ["txttransformer", "textonehot"]:
                    recon_decoded, orig_decoded = output_onehot2text(recon, data[o][0])
                    output = open('{}/visuals/recon_epoch{}_m{}xm{}.txt'.format(runPath, epoch, r, o), "w")
                    joined = []
                    for orig, reconst in zip(orig_decoded, recon_decoded):
                        for l in [orig, "|", reconst, "\n"]:
                            joined.append(l)
                    output.writelines(["".join(joined)])
                    output.close()
                elif self.vaes[o].enc_name.lower() == "audio":
                    _data = torch.stack(data[o]).cpu()[:N]
                    for i in range(3):
                        numpy_to_wav(os.path.join(runPath,"visuals/",'orig_epoch{}_minp{}_s{}.wav'.format(epoch, r, i)),
                                 np.asarray(_data[i].cpu()).astype(np.int16), 16000)
                        numpy_to_wav(os.path.join(runPath,"visuals/",'recon_epoch{}_minp{}_s{}.wav'.format(epoch, r, i)),
                                 np.asarray(recon[i].cpu()).astype(np.int16), 16000)


    def analyse(self, data, runPath, epoch, labels):
        zsl, kls_df = self.analyse_data(data, K=10, runPath=runPath, epoch=epoch, labels=labels)
        plot_kls_df(kls_df, '{}/visuals/kl_distance_{}.png'.format(runPath, epoch))

    def analyse_data(self, data, K, runPath, epoch, labels):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss_sampled = [pz.sample(torch.Size([K, len(data[0])])).view(-1, pz.batch_shape[-1]),
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
        K = 1 if self.modelName == "poe" else K
        t_sne([x.cpu() for x in zss_sampled[1:]], runPath, epoch, K, labels)
        return torch.cat(zsl, 0).cpu().numpy(), kls_df
