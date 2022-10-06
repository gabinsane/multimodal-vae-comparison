import abc
import math

import numpy as np
import torch
from torch.nn import Conv2d, BatchNorm3d, Sequential, TransformerEncoderLayer, Embedding, ReLU, TransformerEncoder, \
    ModuleList, Module, Linear, Dropout, SiLU
import torch.nn.functional as F
from numpy import prod

from models.NetworkTypes import NetworkTypes, NetworkRoles
from models.nn_modules import PositionalEncoding, ConvNet, SamePadConv3d, AttentionResidualBlock
from utils import Constants


class VaeComponent(Module):
    def __init__(self, latent_dim: int, data_dim: tuple, net_type=NetworkTypes.UNSPECIFIED, net_role=NetworkRoles.UNSPECIFIED):
        """
        Base for all

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__()
        self.net_role = net_role
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.net_type = net_type

    @abc.abstractmethod
    def forward(self, x):
        """
            Forward pass

            :param x: data batch
            :type x: list, torch.tensor
            :return: tensor of means, tensor of log variances
            :rtype: tuple(torch.tensor, torch.tensor)
        """
        pass

class VaeEncoder(VaeComponent):
    def __init__(self, latent_dim, data_dim, net_type: NetworkTypes):
        """
        Base for all encoders

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__(latent_dim, data_dim, net_type, net_role=1)
        self.net_role = NetworkRoles.ENCODER


class Enc_CNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim):
        """
        CNN encoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim:
        """
        data_dim = (3, 64, 64)
        super(Enc_CNN, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.CNN)
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.silu = SiLU()
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = 3
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = torch.nn.DataParallel(Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs))
        self.conv2 = torch.nn.DataParallel(Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.conv3 = torch.nn.DataParallel(Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))

        # If input image is 64x64 do fourth convolution
        self.conv_64 = torch.nn.DataParallel(Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs))
        self.pooling = torch.nn.AvgPool2d(kernel_size)
        # Fully connected layers
        self.lin1 = torch.nn.DataParallel(Linear(np.product(self.reshape), hidden_dim))
        self.lin2 = torch.nn.DataParallel(Linear(hidden_dim, hidden_dim))

        # Fully connected layers for mean and variance
        self.mu_layer = torch.nn.DataParallel(Linear(hidden_dim, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(hidden_dim, self.latent_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        if isinstance(x, dict):
            x = x["data"]
        batch_size = x.size(0) if len(x.shape) == 4 else x.size(1)
        # Convolutional layers with ReLu activations
        x = self.silu(self.conv1(x.float()))
        x = self.silu(self.conv2(x))
        x = self.silu(self.conv3(x))
        x = self.silu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = self.silu(self.lin1(x))
        x = (self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_MNIST(VaeEncoder):
    def __init__(self, latent_dim, data_dim):
        """
        Image encoder for the MNIST images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        """
        super(Enc_MNIST, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.FNN)
        self.net_type = "CNN"
        self.hidden_dim = 400
        modules = [Sequential(Linear(784, self.hidden_dim), ReLU(True))]
        modules.extend([Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(True))
                        for _ in range(2 - 1)])
        self.enc = Sequential(*modules)
        self.relu = ReLU()
        self.hidden_mu = Linear(in_features=self.hidden_dim, out_features=latent_dim, bias=True)
        self.hidden_logvar = Linear(in_features=self.hidden_dim, out_features=latent_dim, bias=True)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        h = x.view(*x.size()[:-3], -1)
        h = self.enc(h.float())
        h = h.view(h.size(0), -1)
        mu = self.hidden_mu(h)
        logvar = self.hidden_logvar(h)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


def extra_hidden_layer(hidden_dim):
    return Sequential(Linear(hidden_dim, hidden_dim), ReLU(True))


class Enc_MNISTMoE(VaeEncoder):
    def __init__(self, latent_dim, data_dim, num_hidden_layers=1):
        """
        Encoder for MNIST image data.as originally implemented in https://github.com/iffsid/mmvae

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param num_hidden_layers: how many hidden layers to add
        :type num_hidden_layers: int
        """
        super(Enc_MNISTMoE, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.FNN)
        modules = []
        hidden_dim = 400
        self.net_type = "FNN"
        data_d = int(prod(data_dim))
        modules.append(Sequential(Linear(data_d, hidden_dim), ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.enc = Sequential(*modules)
        self.fc21 = Linear(hidden_dim, latent_dim)
        self.fc22 = Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        e = self.enc(x.view(*x.size()[:-3], -1).float())  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Enc_SVHNMoE(VaeEncoder):
    def __init__(self, latent_dim, data_dim):
        """
        Encoder for SVHN image data.as originally implemented in https://github.com/iffsid/mmvae

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        """
        super(Enc_SVHNMoE, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.FNN)
        imgChans = 3
        fBase = 32
        self.net_type = "CNN"
        self.enc = Sequential(
            # input size: 3 x 32 x 32
            Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            ReLU(True),
            # size: (fBase) x 16 x 16
            Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            ReLU(True),
            # size: (fBase * 2) x 8 x 8
            Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        e = self.enc(x.float())
        lv = self.c2(e).squeeze()
        return self.c1(e).squeeze(), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Enc_SVHN(VaeEncoder):
    def __init__(self, latent_dim, data_dim):
        """
        Image encoder for the SVHN dataset or images 32x32x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        """
        super(Enc_SVHN, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.CNN)
        self.net_type = "CNN"
        self.conv1 = Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv3 = Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv4 = Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1)
        self.relu = ReLU()
        self.hidden_mu = Linear(in_features=128, out_features=latent_dim, bias=True)
        self.hidden_logvar = Linear(in_features=128, out_features=latent_dim, bias=True)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        h = self.conv1(x.float())
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.relu(h)
        h = self.conv4(h)
        h = self.relu(h)
        h = h.view(h.size(0), -1)
        mu = self.hidden_mu(h)
        logvar = self.hidden_logvar(h)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_MNIST_DMVAE(VaeEncoder):
    def __init__(self, latent_dim, data_dim, num_hidden=256, zPrivate_dim=1):
        """
        Encoder for the MNIST dataset with private and shared latent space, source: https://github.com/seqam-lab/DMVAE

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param num_hidden: size of the output features
        :type num_hidden: int
        :param zPrivate_dim: dimensionality of the private latent space
        :type zPrivate_dim: int
        """
        super(Enc_MNIST_DMVAE, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.FNN)
        self.net_type = "FNN"
        temp = 0.66
        self.digit_temp = torch.tensor(temp)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = latent_dim
        self.enc_hidden = Sequential(
            Linear(np.prod(data_dim), num_hidden),
            ReLU())
        self.fc = Linear(num_hidden, 2 * zPrivate_dim + 2 * latent_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        hiddens = self.enc_hidden(x.reshape(1, x.shape[0], -1).float())
        stats = self.fc(hiddens)
        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + Constants.eps)
        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + Constants.eps)
        return [muPrivate, muShared], [stdPrivate, stdShared]


class Enc_SVHN_DMVAE(VaeEncoder):
    def __init__(self, latent_dim, data_dim, zPrivate_dim=4):
        """
        Encoder for the SVHN dataset with private and shared latent space, source: https://github.com/seqam-lab/DMVAE

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param zPrivate_dim: dimensionality of the private latent space
        :type zPrivate_dim: int
        """
        super(Enc_SVHN_DMVAE, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.CNN)
        self.net_type = "CNN"
        temp = 0.66
        self.digit_temp = torch.tensor(temp)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = latent_dim

        self.enc_hidden = Sequential(
            Conv2d(3, 32, 4, 2, 1, bias=False),
            ReLU(),
            Conv2d(32, 64, 4, 2, 1, bias=False),
            ReLU(),
            Conv2d(64, 128, 4, 2, 1, bias=False),
            ReLU(),
            Conv2d(128, 256, 4, 2, 1, bias=False),
            ReLU()
        )

        self.fc = Sequential(
            Linear(256 * 2 * 2, 512),
            ReLU(),
            Dropout(0.1),
            Linear(512, 2 * zPrivate_dim + 2 * latent_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: list of private and shared tensors of means, list of private and shared tensors of log variances
        :rtype: tuple(list, list)
        """
        x = x["data"]
        hiddens = self.enc_hidden(x.float())
        hiddens = hiddens.view(hiddens.size(0), -1)
        stats = self.fc(hiddens)
        stats = stats.unsqueeze(0)
        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + Constants.eps)
        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + Constants.eps)
        return [muPrivate, muShared], [stdPrivate, stdShared]


# Classes
class Enc_FNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim=1):
        """
        Fully connected layer encoder for any type of data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        """
        super(Enc_FNN, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.FNN)
        self.net_type = "FNN"
        self.hidden_dim = 300
        self.lin1 = torch.nn.DataParallel(Linear(np.prod(data_dim), self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(Linear(np.prod(data_dim), self.hidden_dim))
        self.lin3 = torch.nn.DataParallel(Linear(self.hidden_dim, self.hidden_dim))

        self.fc21 = torch.nn.DataParallel(Linear(self.hidden_dim, latent_dim))
        self.fc22 = torch.nn.DataParallel(Linear(self.hidden_dim, latent_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        x = (x).float()
        e = torch.relu(self.lin1(x.view(x.shape[0], -1)))
        e = torch.relu(self.lin2(e))
        e = torch.relu(self.lin3(e))
        lv = self.fc22(e)
        return self.fc21(e), lv


class Enc_Audio(VaeEncoder):
    def __init__(self, latent_dim, data_dim=1):
        """
        Encoder for audio data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        """
        super(Enc_Audio, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.FNN)
        self.net_type = "AudioConv"
        self.latent_dim = latent_dim
        self.TCN = ConvNet(data_dim[0], [128, 128, 96, 96, 64], dropout=0)
        self.mu_layer = Sequential(Linear(64 * data_dim[-1], 32), ReLU(), Linear(32, self.latent_dim))
        self.logvar_layer = Sequential(Linear(64 * data_dim[-1], 32), ReLU(), Linear(32, self.latent_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        inputs = torch.stack(x).cuda() if isinstance(x, list) else x
        output = self.TCN(inputs.float()).permute(0, 2, 1)
        x = output.reshape(inputs.shape[0], -1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        # logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_TransformerIMG(VaeEncoder):
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu"):
        """
        Encoder for a sequence of images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param ff_size: feature dimension of the Transformer
        :type ff_size: int
        :param num_layers: number of Transformer layers
        :type num_layers: int
        :param num_heads: number of Transformer attention heads
        :type num_heads: int
        :param dropout: dropout ofr the Transformer
        :type dropout: float32
        :param activation: activation function
        :type activation: str
        """
        super(Enc_TransformerIMG, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.TRANSFORMER)
        self.net_type = "Transformer"
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.datadim = data_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        # self.conv_pretrained = visnn.models.resnet152(pretrained=True, progress=True)
        hid_channels = 32
        kernel_size = 4
        n_chan = 3
        self.reshape = (hid_channels, kernel_size, kernel_size)
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.convolve = torch.nn.DataParallel(
            Sequential(Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs),
                                torch.SiLU(),
                                Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                torch.SiLU(),
                                Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                torch.SiLU(),
                                Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                torch.SiLU()))
        self.downsample = torch.nn.DataParallel(Linear(np.product(self.reshape), self.latent_dim))
        # Transformer layers
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(TransformerEncoderLayer(d_model=self.latent_dim,
                                                                             nhead=self.num_heads,
                                                                             dim_feedforward=self.ff_size,
                                                                             dropout=self.dropout,
                                                                             activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(
            TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        self.mu_layer = torch.nn.DataParallel(Linear(self.datadim[0] * self.latent_dim, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.datadim[0] * self.latent_dim, self.latent_dim))

    def forward(self, batch):
        """
        Forward pass

        :param batch: list of a data batch and boolean masks for the sequences
        :type batch: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = batch["data"]
        mask = batch["masks"]
        bs, nframes = x.shape[0], x.shape[1]
        imgs_feats = []
        for i in range(x.shape[1]):
            imgs_feats.append(
                self.downsample(self.convolve(x[:, i, :].permute(0, 3, 2, 1).float()).view(-1, np.prod(self.reshape))))
        x = torch.stack(imgs_feats)
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = self.sequence_pos_encoder(x)
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        # extract mu and logvar
        mu = self.mu_layer(final.view(bs, -1))
        logvar = self.logvar_layer(final.view(bs, -1))
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_VideoGPT(VaeEncoder):
    def __init__(self, latent_dim, data_dim=1, n_res_layers=4, downsample=(2, 4, 4)):
        """
        Encoder for image sequences taken from https://github.com/wilson1yan/VideoGPT

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [10, 64, 64, 3] for 64x64x3 image sequences with max length 10 images)
        :type data_dim: list
        :param n_res_layers: number of ResNet layers
        :type n_res_layers: int
        """
        super(Enc_VideoGPT, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.DCNN)
        self.net_type = "3DCNN"
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else latent_dim
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, latent_dim, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, latent_dim, kernel_size=3)
        self.res_stack = Sequential(
            *[AttentionResidualBlock(latent_dim)
              for _ in range(n_res_layers)],
            BatchNorm3d(latent_dim),
            ReLU())
        self.mu_layer = torch.nn.DataParallel(Linear(latent_dim * 16 * 16, latent_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(latent_dim * 16 * 16, latent_dim))

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        h = x.permute(0, 4, 1, 2, 3)
        for conv in self.convs:
            h = F.relu(conv(h.float()))
        h = self.conv_last(h)
        h = self.res_stack(h)
        mu = self.mu_layer(h.view(x.shape[0], -1))
        logvar = self.logvar_layer(h.view(x.shape[0], -1))
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_Transformer(VaeEncoder):
    """ Transformer VAE as implemented in https://github.com/Mathux/ACTOR"""

    def __init__(self, latent_dim, data_dim, ff_size=1024, num_layers=8, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer encoder for arbitrary sequential data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param ff_size: feature dimension of the Transformer
        :type ff_size: int
        :param num_layers: number of Transformer layers
        :type num_layers: int
        :param num_heads: number of Transformer attention heads
        :type num_heads: int
        :param dropout: dropout ofr the Transformer
        :type dropout: float32
        :param activation: activation function
        :type activation: str
        """
        super(Enc_Transformer, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.TRANSFORMER)
        self.net_type = "Transformer"
        self.njoints = data_dim[1]
        self.nfeats = data_dim[2]
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats
        self.mu_layer = torch.nn.DataParallel(Linear(self.latent_dim, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.latent_dim, self.latent_dim))

        self.skel_Embeding = torch.nn.DataParallel(Linear(self.input_feats, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(TransformerEncoderLayer(d_model=self.latent_dim,
                                                                             nhead=self.num_heads,
                                                                             dim_feedforward=self.ff_size,
                                                                             dropout=self.dropout,
                                                                             activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(
            TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))

    def forward(self, batch):
        """
        Forward pass

        :param batch: list of a data batch and boolean masks for the sequences
        :type batch: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = batch["data"]
        mask = batch["masks"]
        bs, nframes, njoints, nfeats = x.shape
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = x.permute((1, 0, 2, 3)).reshape(nframes, bs, njoints * nfeats)
        # embedding of the skeleton
        x = self.skelEmbedding(x.cuda())
        # add positional encoding
        x = self.sequence_pos_encoder(x)
        # transformer layers
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        # get the average of the output
        z = final.mean(axis=0)
        # extract mu and logvar
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_TxtTransformer(VaeEncoder):
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=8, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer encoder configured for character-level text reconstructions

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        """
        super(Enc_TxtTransformer, self).__init__(latent_dim, data_dim, net_type=NetworkTypes.TXTTRANSFORMER)
        self.net_type = "TxtTransformer"
        self.njoints = data_dim[1]
        self.nfeats = data_dim[2]
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.embedding = Embedding(self.input_feats, 2)
        self.sequence_pos_encoder = PositionalEncoding(2, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(TransformerEncoderLayer(d_model=self.input_feats * 2,
                                                                             nhead=self.num_heads,
                                                                             dim_feedforward=self.ff_size,
                                                                             dropout=self.dropout,
                                                                             activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(
            TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        self.mu_layer = torch.nn.DataParallel(Linear(self.input_feats * 2, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.input_feats * 2, self.latent_dim))

    def forward(self, batch):
        """
        Forward pass

        :param batch: list of a data batch and boolean masks for the sequences
        :type batch: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = batch["data"]
        mask = batch["masks"]
        bs, nframes, njoints = x.shape
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = self.embedding(x.long())
        x = self.sequence_pos_encoder(x)
        final = self.seqTransEncoder(x.view(nframes, bs, -1), src_key_padding_mask=~mask)
        z = final.mean(axis=0)
        # extract mu and logvar
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar
