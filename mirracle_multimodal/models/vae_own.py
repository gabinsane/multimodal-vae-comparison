# MNIST model specification

import torch, numpy as np
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import pickle
from utils import Constants, create_vocab, W2V
from vis import plot_embeddings, plot_kls_df
from .vae import VAE
import cv2


# Constants
dataSize = torch.Size([3,64,64])
#data_dim = int(prod(dataSize))


def extra_hidden_layer(hidden_dim=400):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))

# Classes
class Enc2(nn.Module):
    """Parametrizes q(z|x).

    This is the standard DCGAN architecture.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, params, num_hidden_layers=1):
        super(Enc2, self).__init__()
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
        lv = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, lv

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
        return d.squeeze(), torch.tensor(0.75).to(z.device)

# Classes
class Enc(nn.Module):
    """ Generate latent parameters for MNIST image data. """

    def __init__(self, latent_dim, params, num_hidden_layers=1, data_dim=1):
        super(Enc, self).__init__()
        self.hidden_dim = 300
        self.lin1 = torch.nn.DataParallel(nn.Linear(data_dim, self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(data_dim, self.hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.fc21 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, latent_dim))
        self.fc22 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, latent_dim))

    def forward(self, x):
        e = torch.relu(self.lin1(x))
        #e = torch.relu(self.lin2(e))
        #e = torch.relu(self.lin3(e))
        lv = self.fc22(e)
        lv =  F.softmax(lv, dim=-1) + Constants.eta
        return self.fc21(e), lv


class Dec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim, num_hidden_layers=1, data_dim=1):
        super(Dec, self).__init__()
        self.hidden_dim = 20
        self.data_dim = data_dim
        self.lin1 = torch.nn.DataParallel(nn.Linear(latent_dim, self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(latent_dim, self.hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, data_dim))

    def forward(self, z):
        p = torch.relu(self.lin1(z))
        #p = torch.relu(self.lin2(p))
        #p = torch.relu(self.lin3(p))
        d = torch.sigmoid(self.fc3(p))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class UNIVAE(VAE):
    """ Universal VAE used for custom datasets. """

    def __init__(self, params, index):
        if index == 0:
            self.pth = params.mod1
            self.data_dim = params.data_dim1
            self.data_type = params.data1
            if "d.pkl" in self.pth:
                self.num_words = params.num_words1
        elif index == 1:
            self.pth = params.mod2
            self.data_dim = params.data_dim2
            self.data_type = params.data2
            if "d.pkl" in self.pth:
                self.num_words = params.num_words2
        if not ".pkl" in self.pth:
            super(UNIVAE, self).__init__(
                dist.Normal,  # prior
                dist.Normal,  # likelihood
                dist.Normal,  # posterior
                Enc2(params.latent_dim, params, params.num_hidden_layers),
                Dec2(params.latent_dim, params.num_hidden_layers),
                params
            )
        else:
            super(UNIVAE, self).__init__(
                dist.Normal,  # prior
                dist.Normal,  # likelihood
                dist.Normal,  # posterior
                Enc(params.latent_dim, params, params.num_hidden_layers, self.data_dim),
                Dec(params.latent_dim, params.num_hidden_layers, self.data_dim),
                params
            )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'vae_{}'.format(self.data_type)
        self.params = params
        self.noisy = params.noisytxt
        self.llik_scaling = 1
        if "d.pkl" in self.pth:
            self.w2v = W2V(self.data_dim/self.num_words, self.pth)

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=False, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        if not ".pkl" in self.pth:
            d = load_images(self.pth)
            d = d.reshape((d.shape[0], 3, 64, 64))
        else:
            with open(self.pth, 'rb') as handle:
                d = pickle.load(handle)
                if "attrs" in self.pth:
                    if isinstance(d[0][0], str):
                        d, vocab = create_vocab(d, self.noisy)
                else:
                        d = d.reshape(d.shape[0], -1)
                        d = self.w2v.normalize_w2v(d) #* np.random.uniform(low=0.9, high=1.1, size=(d.shape))
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
        samples = super(UNIVAE, self).generate(N, K).cpu().squeeze()
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
        recons_mat = super(UNIVAE, self).reconstruct(data[:8]).squeeze()
        o_l = []
        if "d.pkl" in self.pth:
            _data = data[:8].cpu()
            recon = recons_mat.cpu()
            target, reconstruct = [], []
            _data = _data.reshape(-1, 3, int(self.data_dim / 3))
            recon = recon.reshape(-1, 3, int(self.data_dim / 3))
            for s in _data:
                seq = []
                for w in s:
                    seq.append(self.w2v.model.wv.most_similar(positive=[self.w2v.unnormalize_w2v(np.asarray(w)), ])[0][0])
                target.append(" ".join(seq))
            for s in recon:
                seq = []
                prob = []
                for w in s:
                    seq.append(self.w2v.model.wv.most_similar(positive=[self.w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][0])
                    prob.append("({})".format(str(round(self.w2v.model.wv.most_similar(positive=[self.w2v.unnormalize_w2v(np.asarray(w.cpu())), ])[0][1], 2))))
                j = [" ".join((x, prob[y])) for y,x in enumerate(seq)]
                reconstruct.append(" ".join(j))
            output = open('{}/recon_{:03d}.txt'.format(runPath, epoch), "w")
            output.writelines(["|".join(target) + "\n", "|".join(reconstruct)])
            output.close()
        else:
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
        try:
            zemb, zsl, kls_df = super(UNIVAE, self).analyse(data, K=10)
            labels = ['Prior', self.modelName.lower()]
            plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
            plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
        except:
            pass

def load_images(path, imsize=64):
        import os, glob, numpy as np, imageio
        print("Loading data...")
        images = sorted(glob.glob(os.path.join(path, "*.png")))
        dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
        for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            # image = reshape_image(image, self.imsize)
            # image = cv2.resize(image, (imsize, imsize))
            dataset[i, :] = image / 255
        print("Dataset of shape {} loaded".format(dataset.shape))
        return dataset
