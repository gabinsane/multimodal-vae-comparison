# MNIST-SVHN multi-modal model specification
import os
import cv2
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
from vis import plot_embeddings, plot_kls_df, plot_embeddings_enc, embed_umap
from .mmvae import MMVAE, MMVAE_P
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=3, svd_solver='full')

from .vae_own import UNIVAE

categories = ["sizes", "colors", "shapes"]
map_words = {"large":1, "small":3, "purple":1, "red": 3, "maroon":5, "navy":7, "grey":9, "orange":11, "teal":13, "black":15, "green":17, "blue":19,
             "pieslice":1, "square":3, "semicircle":5, "line": 7, "circle":9}
labels_enc = [["Large (IMG)", "Large (TXT)", "Small (IMG)", "Small (TXT)"],
                  [ "Purple (IMG)","Purple (TXT)", "Red (IMG)", "Red (TXT)",
                    "Maroon (IMG)", "Maroon (TXT)", "Navy (IMG)", "Navy (TXT)",
                    "Grey (IMG)", "Grey (TXT)", "Orange (IMG)", "Orange (TXT)",
                    "Teal (IMG)", "Teal (TXT)", "Black (IMG)", "Black (TXT)",
                    "Green (IMG)", "Green (TXT)", "Blue (IMG)", "Blue (TXT)"],
                  ["Pieslice (IMG)", "Pieslice (TXT)", "Square (IMG)", "Square (TXT)",
                   "Semicircle (IMG)", "Semicircle (TXT)", "Line (IMG)", "Line (TXT)",
                   "Circle (IMG)", "Circle (TXT)"]]
labels_enc2 = [[ "1 (IMG)","1 (TXT)", "2 (IMG)", "2 (TXT)",
                    "3 (IMG)", "3 (TXT)", "4 (IMG)", "4 (TXT)",
                    "5 (IMG)", "5 (TXT)", "6 (IMG)", "6 (TXT)",
                    "7 (IMG)", "7 (TXT)", "8 (IMG)", "8 (TXT)",
                    "9 (IMG)", "9 (TXT)", "10 (IMG)", "10 (TXT)","11 (IMG)","1 (TXT)", "12 (IMG)", "12 (TXT)",
                     "13 (IMG)", "13 (TXT)", "14 (IMG)", "14 (TXT)", "15 (IMG)", "15 (TXT)", "16 (IMG)", "16 (TXT)"]]#              "17 (IMG)", "17 (TXT)", "18 (IMG)", "18 (TXT)",  "19 (IMG)", "19 (TXT)", "20 (IMG)", "21 (TXT)"]]
map_words2 =  {"large purple pieslice":1, "large red pieslice": 3, "large maroon pieslice":5, "large navy pieslice":7,
                            "large grey pieslice":9, "large orange pieslice":11, "large teal pieslice":13, "large black pieslice":15,
                            "large green pieslice":17, "large blue pieslice":19, "large purple square":21, "large red square": 23,
                            "large maroon square":25, "large navy square":27, "large grey square":29, "large orange square":31,
                            "large teal square":33, "large black square":35, "large green square":37, "large blue square":39,
                            "large purple circle":41, "large red circle": 43, "large maroon circle":45, "large navy circle":47,
                            "large grey circle":49, "large orange circle":51, "large teal circle":53, "large black circle":55,
                            "large green circle":57, "large blue circle":59, "large purple semicircle":61, "large red semicircle": 63,
                            "large maroon semicircle":65, "large navy semicircle":67, "large grey semicircle":69, "large orange semicircle":71,
                            "large teal semicircle":73, "large black semicircle":75, "large green semicircle":77, "large blue semicircle":79,
                            "large purple line":81, "large red line": 83, "large maroon line":85, "large navy line":87, "large grey line":89,
                            "large orange line":91, "large teal line":93, "large black line":95, "large green line":97, "large blue line":99,
                            "small purple pieslice":101, "small red pieslice": 103, "small maroon pieslice":105, "small navy pieslice":107,
                            "small grey pieslice":109, "small orange pieslice":111, "small teal pieslice":113, "small black pieslice":115,
                            "small green pieslice":117, "small blue pieslice":119, "small purple square":121, "small red square": 123,
                            "small maroon square":125, "small navy square":127, "small grey square":129, "small orange square":131,
                            "small teal square":133, "small black square":135, "small green square":137, "small blue square":139, "small purple circle":141,
                            "small red circle": 143, "small maroon circle":145, "small navy circle":147, "small grey circle":149,
                            "small orange circle":151, "small teal circle":153, "small black circle":155, "small green circle":157,
                            "small blue circle":159, "small purple semicircle":161, "small red semicircle": 163, "small maroon semicircle":165,
                            "small navy semicircle":167, "small grey semicircle":169, "small orange semicircle":171, "small teal semicircle":173,
                            "small black semicircle":175, "small green semicircle":177, "small blue semicircle":179, "small purple line":181,
                            "small red line": 183, "small maroon line":185, "small navy line":187, "small grey line":189, "small orange line":191,
                            "small teal line":193, "small black line":195, "small green line":197, "small blue line":199}

class MOE(MMVAE):
    def __init__(self, params):
        super(MOE, self).__init__(dist.Normal, params, UNIVAE, UNIVAE)
        grad = {'requires_grad': False}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(params.data_dim2) / prod(params.data_dim1) \
            if int(params["llik_scaling"]) == 0 else int(params["llik_scaling"])
        self.modelName = 'moe-dualmod'

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
                        _data = data[o][3:8].cpu()
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
                                output = open(os.path.join(runPath, 'recon_{}x{}_{}.txt'.format(r, o, epoch)), "w")
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
                        if not (r == 1 and o == 1 and "d.pkl" in self.vaes[o].pth):
                            w2 = cv2.cvtColor(w*255, cv2.COLOR_BGR2RGB)
                            w2 = cv2.copyMakeBorder(w2,top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                            cv2.imwrite(os.path.join(runPath, 'recon_{}x{}_{}.png'.format(r, o, epoch)), w2)
                    except:
                        pass

    def analyse_encodings(self, data, runPath, epoch):
        K = 1
        zemb = []
        qz_xs, px_zs, zss = super(MOE, self).forward(data)
        for qz_x in qz_xs:
            # downsample = pca.fit_transform(qz_x.loc.cpu().detach())
            downsample = TSNE(n_components=3).fit_transform(qz_x.loc.cpu().detach())
            zemb.append(downsample)
        zemb = np.concatenate((zemb[0], zemb[1]))
        sequences = data[1].reshape((data[1].shape[0], 3, -1))
        sizes, cols, shapes = [], [], []
        for s in sequences:
            for _ in range(K):
                sizes.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[0].cpu())), ])[0][0])
                cols.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[1].cpu())), ])[0][0])
                shapes.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[2].cpu())), ])[0][0])
        for ix, word in enumerate([sizes, cols, shapes]):
            word_int = word
            for k in map_words.keys():
                word_int = [x.replace(k, str(map_words[k])) for x in word_int]
            word_int = [int(x) for x in word_int]
            word_int.extend([x+1 for x in word_int])
            plot_embeddings_enc(zemb, word_int, labels_enc[ix], '{}/emb_umap_{}_{}.png'.format(runPath, epoch, categories[ix]))

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
        grad = {'requires_grad': False}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params["n_latents"]), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params["n_latents"]), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(int(params["mod1_datadim"])) / prod(int(params["mod1_datadim"])) \
            if int(params["llik_scaling"]) == 0 else int(params["llik_scaling"])
        self.modelName = 'poe-dualmod'

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


    def analyse_encodings(self, data, runPath, epoch):
        K = 1
        zemb = []
        for m in range(len(data)):
            mods = [None for _ in range(len(data))]
            mods[m] = data[m]
            qz_x, _, z = super(POE, self).forward(mods)
            # samples = qz_x.rsample(torch.Size([K]))
            # samples = np.concatenate([s for s in samples.cpu().detach()])
            # downsample = pca.fit_transform(qz_x.loc.cpu().detach())
            downsample = TSNE(n_components=3).fit_transform(qz_x.loc.cpu().detach())
            zemb.append(downsample)
        zemb = np.concatenate((zemb[0], zemb[1]))
        sequences = data[1].reshape((data[1].shape[0], 3, -1))
        annots = []
        for s in sequences:
            for _ in range(K):
                annot = []
                annot.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[0].cpu())), ])[0][0])
                annot.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[1].cpu())), ])[0][0])
                annot.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[2].cpu())), ])[0][0])
                annots.append(" ".join(annot))
        for k in map_words2.keys():
            annots = [x.replace(k, str(map_words2[k])) for x in annots]
        annots = [int(x) for x in annots]
        annots.extend([x+1 for x in annots])
        plot_embeddings_enc(zemb, annots, labels_enc2[0], '{}/pca_{}_{}.png'.format(runPath, epoch, "whole_inputs"))

    def analyse_encodings_perword(self, data, runPath, epoch):
        K = 1
        zemb = []
        for m in range(len(data)):
            mods = [None for _ in range(len(data))]
            mods[m] = data[m]
            qz_x, _, z = super(POE, self).forward(mods)
            # downsample = pca.fit_transform(qz_x.loc.cpu().detach())
            downsample = TSNE(n_components=3).fit_transform(qz_x.loc.cpu().detach())
            zemb.append(downsample)
        zemb = np.concatenate((zemb[0], zemb[1]))
        sequences = data[1].reshape((data[1].shape[0], 3, -1))
        sizes, cols, shapes = [], [], []
        for s in sequences:
            for _ in range(K):
                sizes.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[0].cpu())), ])[0][0])
                cols.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[1].cpu())), ])[0][0])
                shapes.append(self.vaes[1].w2v.model.wv.most_similar(positive=[self.vaes[1].w2v.unnormalize_w2v(np.asarray(s[2].cpu())), ])[0][0])
        for ix, word in enumerate([sizes, cols, shapes]):
            word_int = word
            for k in map_words.keys():
                word_int = [x.replace(k, str(map_words[k])) for x in word_int]
            word_int = [int(x) for x in word_int]
            word_int.extend([x+1 for x in word_int])
            plot_embeddings_enc(zemb, word_int, labels_enc[ix], '{}/pca_{}_{}.png'.format(runPath, epoch, categories[ix]))

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

