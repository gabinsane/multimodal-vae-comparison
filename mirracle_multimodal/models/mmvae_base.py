# Base MMVAE class definition, common for PoE and MoE
from itertools import combinations
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df
from vis import plot_embeddings, plot_kls_df
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import numpy as np
import cv2

class MMVAE(nn.Module):
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        vae_mods = []
        for ix, vae in enumerate(vaes):
            params["mod_type"] = params["mod{}_type".format(ix + 1)]
            params["mod_path"] = params["mod{}_path".format(ix + 1)] if "mod{}_path".format(
                ix + 1) in params.keys() else ""
            params["mod_numwords"] = params["mod{}_numwords".format(ix + 1)] if "mod{}_numwords".format(
                ix + 1) in params.keys() else ""
            params["mod_datadim"] = int(params["mod{}_datadim".format(ix + 1)]) if "mod{}_datadim".format(
                ix + 1) in params.keys() else ""
            vae_mods.append(vae(params))
        self.vaes = nn.ModuleList(vae_mods)
        self.modelName = None  # filled-in per sub-class
        self.params = params
        grad = {'requires_grad': False}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params["n_latents"]), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params["n_latents"]), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(int(params["mod_datadim"])) / prod(int(params["mod_datadim"])) \
            if int(params["llik_scaling"]) == 0 else int(params["llik_scaling"])

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=False, device='cuda'):
        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)
        train_mnist_svhn = TensorDataset([t1.dataset,t2.dataset])
        test_mnist_svhn = TensorDataset([s1.dataset, s2.dataset])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=False, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=False, **kwargs)
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
                    try:
                        _data = data[o][:8].cpu()
                        if "d.pkl" in self.vaes[o].pth:
                            target, reconstruct = [], []
                            _data = _data.reshape(-1,3,int(self.vaes[o].data_dim/3))
                            recon = recon.reshape(-1, 3, int(self.vaes[o].data_dim/3))
                            for s in _data:
                                seq = []
                                for w in s:
                                    seq.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0])
                                target.append(" ".join(seq))
                            for s in recon:
                                seq = []
                                for w in s:
                                    seq.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0])
                                reconstruct.append(" ".join(seq))
                            output = open(os.path.join(runPath, 'recon_{}x{}_{}.txt'.format(r, o, epoch)), "w")
                            if o == 0:
                                reconstruct = ""
                            if r == 0:
                                target = ""
                            output.writelines(["|".join(target)+"\n", "|".join(reconstruct)])
                            output.close()
                        recon = recon.squeeze(0).cpu()
                        # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                        _data = _data.reshape(-1, 64,64, 3) # if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                        recon = recon.reshape(-1, 64,64, 3) # if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                        o_l = []
                        for x in range(_data.shape[0]):
                            org = cv2.copyMakeBorder(np.asarray(_data[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                            rec = cv2.copyMakeBorder(np.asarray(recon[x]),top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                            if o_l == []:
                                o_l = org
                                r_l = rec
                            else:
                                o_l = np.hstack((o_l, org))
                                r_l = np.hstack((r_l, rec))
                        w = np.vstack((o_l, r_l))
                        if not (r == 1 and o == 1 and "d.pkl" in self.vaes[1].pth):
                            w2 =cv2.cvtColor(w*255, cv2.COLOR_BGR2RGB)
                            w2 = cv2.copyMakeBorder(w2,top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                            cv2.imwrite(os.path.join(runPath, 'recon_{}x{}_{}.png'.format(r, o, epoch)), w2)
                    except:
                        pass


    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = self.analyse_data(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{}.png'.format(runPath, epoch))


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
