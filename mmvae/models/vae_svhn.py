# SVHN model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import numpy as np
from utils import Constants, create_vocab
from vis import plot_embeddings, plot_kls_df
from .vae import VAE
from numpy import prod, sqrt
import pickle
import random
import cv2

# Constants
dataSize = torch.Size([3, 64,64])
imgChans = dataSize[0]
fBase = 32  # base size of filter channels
data_dim = int(prod(dataSize))
hidden_dim = 400



def extra_hidden_layer(hidden_dim = 400):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


class EncT(nn.Module):
    """ Generate latent parameters for CUB image feature. """

    def __init__(self, latent_dim, n_c):
        super(EncT, self).__init__()
        dim_hidden = 8
        self.enc = nn.Sequential()
        self.enc.add_module("layer1", nn.Sequential(
            nn.Linear(n_c, dim_hidden),
            nn.ELU(inplace=True),
        ))
        # relies on above terminating at dim_hidden
        self.fc21 = nn.Linear(dim_hidden, latent_dim)
        self.fc22 = nn.Linear(dim_hidden, latent_dim)

    def forward(self, x):
        e = self.enc(x)
        return self.fc21(e), F.softplus(self.fc22(e)) + Constants.eta


class DecT(nn.Module):
    """ Generate a CUB image feature given a sample from the latent space. """

    def __init__(self, latent_dim, n_c):
        super(DecT, self).__init__()
        self.n_c = n_c
        dim_hidden = 8
        self.dec = nn.Sequential()
        indim = latent_dim
        outdim = 1
        self.dec.add_module("out_t" + "_t", nn.Sequential(
            nn.Linear(indim, outdim),
            nn.ELU(inplace=True),
        ))
        # relies on above terminating at n_c // 2
        self.fc31 = nn.Linear(n_c // 2, n_c)

    def forward(self, z):
        p = self.dec(z.view(-1, z.size(-1)))
        mean = self.fc31(p).view(*z.size()[:-1], -1)
        return mean, torch.tensor([0.01]).to(mean.device)


# Classes
class Enc2(nn.Module):
    """Parametrizes q(z|x).

    This is the standard DCGAN architecture.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Enc2  , self).__init__()
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = 3

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = torch.nn.DataParallel(nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs))
        self.conv2 = torch.nn.DataParallel(nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.conv3 = torch.nn.DataParallel(nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))

        # If input image is 64x64 do fourth convolution
        self.conv_64 = torch.nn.DataParallel(nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))

        # Fully connected layers
        self.lin1 = torch.nn.DataParallel(nn.Linear(np.product(self.reshape), hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(hidden_dim, hidden_dim))

        # Fully connected layers for mean and variance
        self.mu_gen = torch.nn.DataParallel(nn.Linear(hidden_dim, self.latent_dim))
        self.var_gen = torch.nn.DataParallel(nn.Linear(hidden_dim, self.latent_dim))

    def forward(self, x):
        batch_size = x.size(0) if len(x.shape) == 4 else x.size(1)
        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1 ))
        x = torch.relu(self.lin1(x))
        x = (self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu = self.mu_gen(x)
        logvar = self.var_gen(x)
        return mu, F.softmax(logvar, dim=-1) * logvar.size(-1) + Constants.eta

class Dec2(nn.Module):
    """Parametrizes p(x|z).

    This is the standard DCGAN architecture.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Dec2, self).__init__()
        latent_dim = latent_dim

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        self.n_chan = 3

        # Fully connected lay
        self.lin1 = torch.nn.DataParallel(nn.Linear(latent_dim, hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(hidden_dim, hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(hidden_dim, np.product(self.reshape)))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        self.convT_64 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))

        self.convT1 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.convT2 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.convT3 = torch.nn.DataParallel(nn.ConvTranspose2d(hid_channels, self.n_chan, kernel_size, **cnn_kwargs))

    def forward(self, z):
        if len(z.shape) == 2:
            batch_size = z.size(0)
        else:
            batch_size = z.size(1)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size * x.shape[0], *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = (self.convT3(x))
        d = torch.sigmoid(x.view(*z.size()[:-1], *dataSize))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)



class Enc(nn.Module):
    """ Generate latent parameters for MNIST image data. """

    def __init__(self, latent_dim, num_hidden_layers=1, data_dim=1):
        super(Enc, self).__init__()
        modules = []
        if data_dim < 200:
            self.hidden_dim = int(data_dim * 2)
        elif data_dim == 256:
            self.hidden_dim = 256
        else:
            self.hidden_dim = int(data_dim/2)
        modules.append(nn.Sequential(nn.Linear(data_dim, self.hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(self.hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(self.hidden_dim, latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x)  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Dec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim, num_hidden_layers=1, data_dim=1):
        super(Dec, self).__init__()
        self.data_dim = data_dim
        if data_dim < 200:
            self.hidden_dim = int(data_dim * 2)
        elif data_dim == 256:
            self.hidden_dim = 256
        else:
            self.hidden_dim = int(data_dim/2)
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, self.hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(self.hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, self.data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], self.data_dim))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class SVHN(VAE):
    """ Derive a specific sub-class of a VAE for SVHN """

    def __init__(self, params):
        super(SVHN, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,  # posterior
            Enc(params.latent_dim),
            Dec(params.latent_dim),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'svhn'
        self.dataSize = dataSize
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()
        train = DataLoader(datasets.SVHN('../data', split='train', download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN('../data', split='test', download=True, transform=tx),
                          batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(SVHN, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(SVHN, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(SVHN, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

class CROW2(VAE):
    """ Derive a specific sub-class of a VAE for SVHN """

    def __init__(self, params):
        self.noisy = params.noisytxt
        self.pth = params.mod2
        if not ".pkl" in params.mod2:
            super(CROW2, self).__init__(
                dist.Normal,  # prior
                dist.Normal,  # likelihood
                dist.Normal,  # posterior
                Enc2(params.latent_dim),
                Dec2(params.latent_dim),
                params
            )
        else:
            # super(CROW2, self).__init__(
            #     dist.Normal,  # prior
            #     dist.Normal,  # likelihood
            #     dist.Normal,  # posterior
            #     EncT(params.latent_dim, 2),
            #     DecT(params.latent_dim, 2),
            #     params
            # )
            super(CROW2, self).__init__(
                dist.Normal,  # prior
                dist.Normal,  # likelihood
                dist.Normal,  # posterior
                Enc(params.latent_dim, 1, params.data_dim2),
                Dec(params.latent_dim, 1, params.data_dim2),
                params
            )

        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'imagetxt'
        self.dataSize = dataSize
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=False, device='cuda'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()

        if not ".pkl" in self.pth:
            d = load_images(self.pth)
            data_size = d.shape[0]
            d = d.reshape((data_size, 3, 64, 64))
        else:
            with open(self.pth, 'rb') as handle:
                d = pickle.load(handle)
                d = np.expand_dims(d, axis=1).squeeze()
                #d, vocab = create_vocab(d, self.noisy)
                if "attrs" in self.pth:
                    d = d[:,0]
                    d = np.expand_dims(d, axis=1)
                    d, vocab = create_vocab(d, self.noisy)
                if len(d.shape) < 2:
                    d = np.expand_dims(d, axis=1)
                data_size = d.shape[0]
        traindata = torch.Tensor(d[:int(data_size*(0.9))])
        testdata = torch.Tensor(d[int(data_size*(0.9)):])
        crow_t_dataset = torch.utils.data.TensorDataset(traindata)
        crow_v_dataset = torch.utils.data.TensorDataset(testdata)
        train = DataLoader(crow_t_dataset,
                           batch_size=batch_size, shuffle=False, **kwargs)
        test = DataLoader(crow_v_dataset,
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N, K = 36, 1
        samples = super(CROW2, self).generate(N, K).cpu().squeeze()
        # wrangle things so they come out tiled
        #samples = samples.view(N, *samples.size()[1:])
        r_l = []
        for r, recons_list in enumerate(samples):
                recon = recons_list.cpu()
                recon = recon.reshape(64,64,3).unsqueeze(0)
                if r_l == []:
                        r_l = np.asarray(recon)
                else:
                        r_l = np.concatenate((r_l, np.asarray(recon)))
        r_l = np.vstack((np.hstack(r_l[:6]), np.hstack(r_l[6:12]), np.hstack(r_l[12:18]), np.hstack(r_l[18:24]),  np.hstack(r_l[24:30]),  np.hstack(r_l[30:36])))
        cv2.imwrite('{}/gen_samples_{:03d}.png'.format(runPath, epoch), r_l*255)

    def reconstruct(self, data, runPath, epoch):
        recons_mat = super(CROW2, self).reconstruct(data[:8]).squeeze()
        o_l = []
        for r, recons_list in enumerate(recons_mat):
                _data = data[r].cpu()
                recon = recons_list.cpu()
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                _data = _data.reshape(-1, 64,64, 3) # if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                recon = recon.reshape(-1, 64,64, 3) # if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                if o_l == []:
                        o_l = np.asarray(_data)
                        r_l = np.asarray(recon)
                else:
                        o_l = np.concatenate((o_l, np.asarray(_data)))
                        r_l = np.concatenate((r_l, np.asarray(recon)))
        w = np.vstack((np.hstack(o_l), np.hstack(r_l)))
        cv2.imwrite('{}/recon_{}x_{:03d}.png'.format(runPath, r, epoch), w*255)

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(CROW2, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))


def load_images(path, imsize=64):
        import os, glob, numpy as np, imageio
        print("Loading data...")
        images = sorted(glob.glob(os.path.join(path, "*.png")))
        dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
        for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            # image = reshape_image(image, self.imsize)
            #image = cv2.resize(image, (imsize, imsize))
            dataset[i, :] = image / 255
        print("Dataset of shape {} loaded".format(dataset.shape))
        return dataset



