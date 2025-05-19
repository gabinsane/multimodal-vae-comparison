import abc
import math
import numpy as np
import torch
from torch.nn import Conv2d, BatchNorm3d, Sequential, TransformerEncoderLayer, Embedding, ReLU, TransformerEncoder, \
    ModuleList, Module, Linear, SiLU
import torch.nn.functional as F
from numpy import prod
from models.NetworkTypes import NetworkTypes, NetworkRoles
from models.nn_modules import PositionalEncoding, ConvNet, SamePadConv3d, AttentionResidualBlock, expand_layer, ResUp, ResDown
from models.nn_modules import PositionalEncoding, ConvNet, SamePadConv3d, AttentionResidualBlock, Flatten
from utils import Constants
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16

class VaeComponent(Module):
    def __init__(self, latent_dim: int, data_dim: tuple, latent_private=None, enc_mu_logvar=True, net_type=NetworkTypes.UNSPECIFIED, net_role=NetworkRoles.UNSPECIFIED):
        """
        Base for all

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__()
        self.net_role = net_role
        self.latent_dim = latent_dim
        self.enc_mu_logvar = enc_mu_logvar
        self.latent_private = latent_private
        if self.latent_private is not None:
            self.out_dim = self.latent_dim + self.latent_private
        else:
            self.out_dim = self.latent_dim
        self.data_dim = data_dim
        self.net_type = net_type
        self.mu_layer = None
        self.logvar_layer = None

    def init_final_layers(self, in_feats):
        self.mu_layer = Linear(in_feats, self.out_dim, bias=True)
        if self.enc_mu_logvar:
            self.logvar_layer = Linear(in_feats, self.out_dim, bias=True)


    def process_output(self, data, inter_outputs=None):
        out_mus = self.mu_layer(data)
        if self.enc_mu_logvar:
            out_lvs = F.softmax(self.logvar_layer(data), dim=-1) + Constants.eta
            return out_mus, out_lvs
        return out_mus

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
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar, net_type: NetworkTypes):
        """
        Base for all encoders

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type, net_role=1)
        self.net_role = NetworkRoles.ENCODER


class Enc_CNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        CNN encoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim:
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        data_dim = (3, 64, 64)
        super(Enc_CNN, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.CNN)
        hid_channels = 32
        kernel_size = 4
        self.relu = ReLU()
        self.hidden_dim = 512
        self.silu = SiLU()
        self.o_shapes = [(32, 32, 32), (32, 16, 16), (32, 8, 8), (512), (256), (256)]
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Fully connected layers for mean and variance
        self.hidden_dim = 1000
        self.mu_layer = Linear(self.hidden_dim, self.out_dim)
        self.logvar_layer = Linear(self.hidden_dim, self.out_dim)
        self.init_final_layers(self.hidden_dim)

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
        out = self.silu(self.resnet(x))
        return self.process_output(out)


class Enc_VIT(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Vision Transformer (ViT) encoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim:
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        data_dim = (3, 64, 64)
        super(Enc_VIT, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.CNN)
        self.silu = SiLU()
        self.vit = vit_b_16(image_size=64)
        self.hidden_dim = 1000
        self.init_final_layers(self.hidden_dim)

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
        out = self.silu(self.vit(x))
        return self.process_output(out)

class Enc_CNN2(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        CNN encoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim:
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        data_dim = (3, 64, 64)
        super(Enc_CNN2, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.CNN)
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.relu = ReLU()
        self.hidden_dim = 512
        self.silu = SiLU()
        self.o_shapes = [(32, 32, 32), (32, 16, 16), (32, 8, 8), (512), (256), (256)]
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = 3
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # # If input image is 64x64 do fourth convolution
        self.conv4 = Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        # self.pooling = torch.nn.AvgPool2d(kernel_size)
        # Fully connected layers
        self.lin1 = Linear(np.product(self.reshape), self.hidden_dim)
        #self.lin2 = Linear(self.hidden_dim, self.hidden_dim)
        #self.net_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.lin1, self.lin2]
        self.init_final_layers(self.hidden_dim)

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
        o1 = self.silu(self.conv1(x.float()))
        o2 = self.silu(self.conv2(o1))
        o3 = self.silu(self.conv3(o2))
        o4 = self.silu(self.conv4(o3))
        #
        # # Fully connected layers with ReLu activations
        o4 = o4.view((batch_size, -1))
        o5 = self.lin1(o4)
        # o6 = (self.lin2(o5))
        return self.process_output(o5)


class Enc_MNIST(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Image encoder for the MNIST images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_MNIST, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.FNN)
        self.net_type = "CNN"
        self.hidden_dim = 400
        modules = [Sequential(Linear(784, self.hidden_dim), ReLU(True))]
        modules.extend([Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(True))
                        for _ in range(2 - 1)])
        self.enc = Sequential(*modules)
        self.relu = ReLU()
        self.hidden_mu = Linear(in_features=self.hidden_dim, out_features=self.out_dim, bias=True)
        self.hidden_logvar = Linear(in_features=self.hidden_dim, out_features=self.out_dim, bias=True)

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


class Enc_RESCNN(VaeEncoder):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n
    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """

    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        super(Enc_RESCNN, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar,  net_type=NetworkTypes.CNN)
        channels = 3
        ch = 64
        self.conv_in = torch.nn.Conv2d(channels, ch, 7, 1, 3)
        self.res_down_block1 = ResDown(ch, 2 * ch)
        self.res_down_block2 = ResDown(2 * ch, 4 * ch)
        self.res_down_block3 = ResDown(4 * ch, 8 * ch)
        self.res_down_block4 = ResDown(8 * ch, 16 * ch)
        self.mu_layer = torch.nn.Conv2d(16 * ch, latent_dim, 4, 1)
        self.logvar_layer = torch.nn.Conv2d(16 * ch, latent_dim, 4, 1)
        self.act_fnc = torch.nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x["data"]))
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 16
        x = self.res_down_block3(x)  # 8
        x = self.res_down_block4(x)  # 4
        return [o.squeeze() for o in self.process_output(x)]


class Enc_MNISTMoE(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar,  num_hidden_layers=1):
        """
        Encoder for MNIST image data.as originally implemented in https://github.com/iffsid/mmvae

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param num_hidden_layers: how many hidden layers to add
        :type num_hidden_layers: int
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_MNIST2, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.FNN)
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

class Enc_PolyMNIST(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Encoder for PolyMNIST image data.as originally implemented in https://github.com/gr8joo/MVTCAE/blob/master/mmnist/networks/ConvNetworksImgCMNIST.py

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_PolyMNIST, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.FNN)
        hidden_dim = 400
        self.net_type = "CNN"
        self.encoder = torch.nn.Sequential(                          # input shape (3, 28, 28)
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            torch.nn.ReLU(),
            Flatten(),                                                # -> (2048)
            torch.nn.Linear(2048, hidden_dim),
            torch.nn.ReLU(),
        )
        self.fc21 = Linear(hidden_dim, self.out_dim)
        self.fc22 = Linear(hidden_dim, self.out_dim)

    def forward(self, x):
        """
        Forward pass

        :param x: data batch
        :type x: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = x["data"]
        e = self.encoder(x.float())
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class Enc_SVHN2(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Encoder for SVHN image data.as originally implemented in https://github.com/iffsid/mmvae

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_SVHN2, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.FNN)
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
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Image encoder for the SVHN dataset or images 32x32x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_SVHN, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.CNN)
        self.net_type = "CNN"
        self.conv1 = Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv3 = Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv4 = Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1)
        self.relu = ReLU()
        self.hidden_mu = Linear(in_features=128, out_features=self.out_dim, bias=True)
        self.hidden_logvar = Linear(in_features=128, out_features=self.out_dim, bias=True)

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

class Enc_FNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Fully connected layer encoder for any type of data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_FNN, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.FNN)
        self.net_type = "FNN"
        self.hidden_dim = 128
        self.lin1 = torch.nn.DataParallel(Linear(np.prod(data_dim), self.hidden_dim))
        self.init_final_layers(self.hidden_dim)

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
        #e = torch.relu(self.lin2(e))
        #e = torch.relu(self.lin3(e))
        return self.process_output(e)


class Enc_TransformerIMG(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu"):
        """
        Encoder for a sequence of images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
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
        super(Enc_TransformerIMG, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.TRANSFORMER)
        self.net_type = "Transformer"
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
        self.downsample = torch.nn.DataParallel(Linear(np.product(self.reshape), self.out_dim))
        # Transformer layers
        self.sequence_pos_encoder = PositionalEncoding(self.out_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(TransformerEncoderLayer(d_model=self.out_dim,
                                                                             nhead=self.num_heads,
                                                                             dim_feedforward=self.ff_size,
                                                                             dropout=self.dropout,
                                                                             activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(
            TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        self.mu_layer = torch.nn.DataParallel(Linear(self.datadim[0] * self.out_dim, self.out_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.datadim[0] * self.out_dim, self.out_dim))

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
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar, n_res_layers=4, downsample=(2, 4, 4)):
        """
        Encoder for image sequences taken from https://github.com/wilson1yan/VideoGPT

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [10, 64, 64, 3] for 64x64x3 image sequences with max length 10 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param n_res_layers: number of ResNet layers
        :type n_res_layers: int
        """
        super(Enc_VideoGPT, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.DCNN)
        self.net_type = "3DCNN"
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else self.out_dim
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, self.out_dim, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, self.out_dim, kernel_size=3)
        self.res_stack = Sequential(
            *[AttentionResidualBlock(self.out_dim)
              for _ in range(n_res_layers)],
            BatchNorm3d(self.out_dim),
            ReLU())
        self.mu_layer = torch.nn.DataParallel(Linear(self.out_dim * 16 * 16 * 4, self.out_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.out_dim * 16 * 16 * 4, self.out_dim))

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
        mu = self.mu_layer(h.reshape(x.shape[0], -1))
        logvar = self.logvar_layer(h.reshape(x.shape[0], -1))
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_Transformer(VaeEncoder):
    """ Transformer VAE as implemented in https://github.com/Mathux/ACTOR"""

    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar, ff_size=1024, num_layers=8, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer encoder for arbitrary sequential data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
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
        super(Enc_Transformer, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.TRANSFORMER)
        self.net_type = "Transformer"
        self.njoints = data_dim[1]
        self.nfeats = data_dim[2]

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats
        self.mu_layer = torch.nn.DataParallel(Linear(self.out_dim, self.out_dim))
        self.logvar_layer = torch.nn.DataParallel(Linear(self.out_dim, self.out_dim))

        self.skel_Embedding = torch.nn.DataParallel(Linear(self.input_feats, self.out_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.out_dim, self.dropout)
        seqTransEncoderLayer = (TransformerEncoderLayer(d_model=self.out_dim,
                                                                             nhead=self.num_heads,
                                                                             dim_feedforward=self.ff_size,
                                                                             dropout=self.dropout,
                                                                             activation=self.activation))
        self.seqTransEncoder = (
            TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))

    def forward(self, batch):
        """
        Forward pass

        :param batch: list of a data batch and boolean masks for the sequences
        :type batch: list, torch.tensor
        :return: tensor of means, tensor of log variances
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = batch["data"].double()
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        mask = batch["masks"]
        bs, nframes, njoints, nfeats = x.shape
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = x.permute((1, 0, 2, 3)).reshape(nframes, bs, njoints * nfeats)
        # embedding of the skeleton
        x = self.skel_Embedding(x.cuda().float())
        # add positional encoding
        x = self.sequence_pos_encoder(x)
        # transformer layers
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        # get the average of the output
        z = final.mean(axis=0)
        return self.process_output(z)


class Enc_ConvTxt(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar):
        """
        Encoder configured for text reconstructions

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_ConvTxt, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar,  net_type=NetworkTypes.TXTTRANSFORMER)
        self.embedding = torch.nn.Embedding(data_dim[-1], 32, padding_idx=0)
        fBase = 32
        self.enc = torch.nn.Sequential(
            # input size: 1 x 32 x 128
            torch.nn.Conv2d(1, fBase, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(fBase),
            torch.nn.ReLU(True),
            # size: (fBase) x 16 x 64
            torch.nn.Conv2d(fBase, fBase * 2, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(fBase * 2),
            torch.nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            torch.nn.Conv2d(fBase * 2, fBase * 3, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(fBase * 3),
            torch.nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            torch.nn.Conv2d(fBase * 3, fBase * 3, (1, 3), (1, 2), (0, 1), bias=False),
            torch.nn.BatchNorm2d(fBase * 3),
            torch.nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            torch.nn.Conv2d(fBase * 3, fBase * 3, (1, 3), (1, 2), (0, 1), bias=False),
            torch.nn.BatchNorm2d(fBase * 3),
            torch.nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        #self.c1 = torch.nn.Conv2d(fBase * 3, self.out_dim, (1,4), 1, 0, bias=False)
        #self.c2 = torch.nn.Conv2d(fBase * 3, self.out_dim, (1,4), 1, 0, bias=False)
        self.mu_layer = torch.nn.Linear(fBase * 3, self.out_dim)
        self.logvar_layer = torch.nn.Linear(fBase * 3, self.out_dim)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, batch):
        x = torch.argmax(batch["data"], dim=-1)
        emb = self.embedding(x).unsqueeze(1)
        e = self.enc(emb)
        mu = self.mu_layer(e.squeeze().view(x.shape[0],-1))
        logvar = self.logvar_layer(e.squeeze().view(x.shape[0], -1))
        return mu, F.softplus(logvar) + Constants.eta
        # extract mu and logvar
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_TxtTransformer(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar, ff_size=128, num_layers=1, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer encoder configured for character-level text reconstructions

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Enc_TxtTransformer, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar, net_type=NetworkTypes.TXTTRANSFORMER)
        self.net_type = "TxtTransformer"
        self.njoints = data_dim[-1]
        self.nfeats = data_dim[-2]

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.hidden_dim = self.out_dim

        self.input_feats = self.njoints * self.nfeats
        self.embedding_size = 2
        self.embedding = torch.nn.Embedding(self.input_feats, self.embedding_size)
        self.sequence_pos_encoder = PositionalEncoding(self.embedding_size, self.dropout)
        seqTransEncoderLayer = (torch.nn.TransformerEncoderLayer(d_model=self.input_feats * self.embedding_size,
                                                                                nhead=self.num_heads,
                                                                                dim_feedforward=self.ff_size,
                                                                                dropout=self.dropout,
                                                                                activation=self.activation))
        self.seqTransEncoder = (
            torch.nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        self.mu_layer = torch.nn.DataParallel(torch.nn.Linear(self.input_feats * self.embedding_size, self.out_dim))
        self.logvar_layer = torch.nn.DataParallel(torch.nn.Linear(self.input_feats * self.embedding_size, self.out_dim))

    def forward(self, batch):
        x = batch["data"]
        mask = batch["masks"]
        bs, nframes, njoints = x.shape
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = self.embedding(x.cuda().long())
        x = self.sequence_pos_encoder(x)
        final = self.seqTransEncoder(x.view(nframes, bs, -1), src_key_padding_mask=~mask)
        z = final.mean(axis=0)
        return self.process_output(z)


class Enc_TxtRNN(VaeEncoder):
    def __init__(self, latent_dim, data_dim, latent_private, enc_mu_logvar, hidden_size=512, n_layers=1, bidirectional=True):
        super(Enc_TxtRNN, self).__init__(latent_dim, data_dim, latent_private, enc_mu_logvar,  net_type=NetworkTypes.TXTTRANSFORMER)
        self.input_size = data_dim[-1] * data_dim[-2]
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embed = torch.nn.Embedding(self.input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.1, bidirectional=bidirectional)
        self.o2p = torch.nn.Linear(hidden_size, self.out_dim * 2)

    def forward(self, batch):
        x = batch["data"]
        mask = batch["masks"]
        embedded = self.embed(x.cuda().long()).unsqueeze(1)

        output, hidden = self.gru(embedded, None)
        # mean loses positional info?
        # output = torch.mean(output, 0).squeeze(0) #output[-1] # Take only the last value
        output = output[-1]  # .squeeze(0)
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[:, self.hidden_size:]  # Sum bidirectional outputs
        else:
            output = output[:, :self.hidden_size]

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar