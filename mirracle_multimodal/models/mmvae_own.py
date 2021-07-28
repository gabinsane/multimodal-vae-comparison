# MNIST-SVHN multi-modal model specification
import os
import cv2
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid
from vis import plot_embeddings, plot_kls_df
from .mmvae import MMVAE, MMVAE_P
from .vae_mnist import MNIST, CROW
from .vae_svhn import SVHN, CROW2
from .vae_own import UNIVAE

class MOE(MMVAE):
    def __init__(self, params):
        super(MOE, self).__init__(dist.Normal, params, UNIVAE, UNIVAE)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(params.data_dim2) / prod(params.data_dim1) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'moe-dualmod'
        self.imgpath = params.mod1
        self.txtpath = params.mod2

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

    def generate(self, runPath, epoch):
        N = 36
        samples_list = super(MOE, self).generate(N)
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

    def reconstruct(self, data, runPath, epoch):
        recons_mat = super(MOE, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                    try:
                        _data = data[o][:8].cpu()
                        if "d.pkl" in self.vaes[o].pth and o == 1:
                            target, reconstruct = [], []
                            _data = _data.reshape(-1,3,int(self.vaes[o].data_dim/3))
                            recon = recon.reshape(-1, 3, int(self.vaes[o].data_dim/3))
                            for s in _data:
                                seq = []
                                for w in s:
                                    seq.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w)), ])[0][0])
                                target.append(" ".join(seq))
                            for s in recon:
                                seq = []
                                for w in s:
                                    seq.append(self.vaes[o].w2v.model.wv.most_similar(positive=[self.vaes[o].w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0])
                                reconstruct.append(" ".join(seq))
                            if o == 0:
                                reconstruct = ""
                            if r == 0:
                                target = ""
                            if not (o == 0 and r ==0):
                                output = open('{}/recon_{}x{}_{}.txt'.format(runPath, r, o, epoch), "w")
                                output.writelines(["|".join(target)+"\n", "|".join(reconstruct)])
                                output.close()
                        recon = recon.squeeze(0).cpu()
                        # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                        _data = _data.reshape(-1, 64,64, 3) # if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                        recon = recon.reshape(-1, 64,64, 3) # if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                        o_l = []
                        for x in range(_data.shape[0]):
                            if o_l == []:
                                o_l = np.asarray(_data[x])
                                r_l = np.asarray(recon[x])
                            else:
                                o_l = np.hstack((o_l, np.asarray(_data[x])))
                                r_l = np.hstack((r_l, np.asarray(recon[x])))
                        w = np.vstack((o_l, r_l))
                        if not (r == 1 and o == 1 and "d.pkl" in self.vaes[o].pth):
                            w2 =cv2.cvtColor(w*255, cv2.COLOR_BGR2RGB)
                            cv2.imwrite('{}/recon_{}x{}_{}.png'.format(runPath, r, o, epoch), w2)
                    except:
                        pass

    def analyse(self, data, runPath, epoch):
        try:
            zemb, zsl, kls_df = super(MOE, self).analyse(data, K=10)
            labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
            plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{}.png'.format(runPath, epoch))
            plot_kls_df(kls_df, '{}/kl_distance_{}.png'.format(runPath, epoch))
        except:
            pass


def resize_img(img, refsize):
    #return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
    return img


class POE(MMVAE_P):
    def __init__(self, params):
        super(POE, self).__init__(dist.Normal, params, UNIVAE, UNIVAE)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(params.data_dim2) / prod(params.data_dim1) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'poe-dualmod'
        self.imgpath = params.mod1
        self.txtpath = params.mod2

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


    def generate(self, runPath, epoch):
        N = 36
        samples_list = super(POE, self).generate(N)
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

    def reconstruct(self, data, runPath, epoch):
        recons_mat = []
        for ix, i in enumerate(data):
            input_mat = [None, None]
            input_mat[ix] = i[:8]
            rec = super(POE, self).reconstruct(input_mat)
            recons_mat.append(rec)
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
                            output = open('{}/recon_{}x{}_{}.txt'.format(runPath, r, o, epoch),"w")
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
                            if o_l == []:
                                o_l = np.asarray(_data[x])
                                r_l = np.asarray(recon[x])
                            else:
                                o_l = np.hstack((o_l, np.asarray(_data[x])))
                                r_l = np.hstack((r_l, np.asarray(recon[x])))
                        w = np.vstack((o_l, r_l))
                        if not (r == 1 and o == 1 and "d.pkl" in self.vaes[1].pth):
                            w2 =cv2.cvtColor(w*255, cv2.COLOR_BGR2RGB)
                            cv2.imwrite('{}/recon_{}x{}_{}.png'.format(runPath, r, o, epoch), w2)
                    except:
                        pass

    def analyse(self, data, runPath, epoch):
        try:
            zemb, zsl, kls_df = super(POE, self).analyse(data, K=10)
            labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
            plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{}.png'.format(runPath, epoch))
            plot_kls_df(kls_df, '{}/kl_distance_{}.png'.format(runPath, epoch))
        except:
           pass

def resize_img(img, refsize):
    #return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
    return img

