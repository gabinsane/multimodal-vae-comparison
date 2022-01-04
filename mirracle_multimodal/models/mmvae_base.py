# Base MMVAE class definition, common for PoE and MoE
from itertools import combinations
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df, plot_embeddings, plot_kls_df
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import numpy as np
import cv2
from .vae import VAE

class MMVAE(nn.Module):
    def __init__(self, prior_dist, encoders, decoders, data_paths, feature_dims, n_latents):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        vae_mods = []
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

    def getDataLoaders(self, batch_size, device='cuda'):
        trains, tests = [], []
        for x in range(len(self.vaes)):
            t, v = self.vaes[x].getDataLoaders(batch_size, device)
            trains.append(t.dataset)
            tests.append(v.dataset)
        train_data = TensorDataset(trains)
        test_data = TensorDataset(tests)
        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
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

    def generate(self, runPath, epoch):
        N = 36
        samples_list = self.generate_samples(N)
        for i, samples in enumerate(samples_list):
            if samples.shape[0] != N:
                samples = samples.reshape(N, 64,64,3)
            try:
                r_l = []
                for r, recons_list in enumerate(samples):
                    recon = recons_list.cpu()
                    recon = recon.reshape(64, 64, 3).unsqueeze(0)
                    if r_l == []:
                        r_l = np.asarray(recon)
                    else:
                        r_l = np.concatenate((r_l, np.asarray(recon)))
                r_l = np.vstack((np.hstack(r_l[:6]), np.hstack(r_l[6:12]), np.hstack(r_l[12:18]), np.hstack(r_l[18:24]),
                                 np.hstack(r_l[24:30]), np.hstack(r_l[30:36])))
                cv2.imwrite('{}/gen_samples_{}_{}.png'.format(runPath, i, epoch), r_l * 255)
            except:
                pass

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons

    def process_reconstructions(self, recons_mat, data, epoch, runPath):
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[o][:8].cpu()
                if "d.pkl" in self.vaes[o].pth:
                    target, reconstruct = [], []
                    _data = _data.reshape(-1,self.vaes[o].data_dim[1], self.vaes[o].data_dim[0])
                    recon = recon.reshape(-1, self.vaes[o].data_dim[1], self.vaes[o].data_dim[0])
                    for d, rec in zip(_data, recon):
                        seq_d, seq_r = [], []
                        [seq_d.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0]) for w in d]
                        [seq_r.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0]) for w in rec]
                        target.append(" ".join(seq_d))
                        reconstruct.append(" ".join(seq_r))
                    output = open(os.path.join(runPath, 'recon_{}x{}_{}.txt'.format(r, o, epoch)), "w")
                    if o == 0: reconstruct = ""
                    if r == 0: target = ""
                    output.writelines(["|".join(target)+"\n", "|".join(reconstruct)])
                    output.close()
                else:
                    recon = recon.squeeze(0).cpu()
                    _data = _data.reshape(-1, 64,64, 3)
                    recon = recon.reshape(-1, 64,64, 3)
                    o_l, r_l = [], []
                    for x in range(_data.shape[0]):
                        org = cv2.copyMakeBorder(np.asarray(_data[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                        rec = cv2.copyMakeBorder(np.asarray(recon[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                        o_l = org if o_l == [] else np.hstack((o_l, org))
                        r_l = rec if r_l == [] else np.hstack((r_l, rec))
                    w = np.vstack((o_l, r_l))
                if not (r == 1 and o == 1 and "d.pkl" in self.vaes[1].pth):
                    w2 =cv2.cvtColor(w*255, cv2.COLOR_BGR2RGB)
                    w2 = cv2.copyMakeBorder(w2,top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                    cv2.imwrite(os.path.join(runPath, 'recon_{}x{}_{}.png'.format(r, o, epoch)), w2)


    def analyse(self, data, runPath, epoch):
        try:
            zemb, zsl, kls_df = self.analyse_data(data, K=10)
            labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
            plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{}.png'.format(runPath, epoch))
            plot_kls_df(kls_df, '{}/kl_distance_{}.png'.format(runPath, epoch))
        except:
            pass


    def analyse_data(self, data, K):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df
