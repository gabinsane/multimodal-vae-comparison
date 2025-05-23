import math
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod

from models.NetworkTypes import NetworkTypes, NetworkRoles
from models.encoders import VaeComponent
from models.nn_modules import PositionalEncoding, AttentionResidualBlock, SamePadConvTranspose3d, Unflatten
from models.nn_modules import DeconvNet, expand_layer, ResUp
from models.nn_modules import PositionalEncoding, AttentionResidualBlock, \
    SamePadConvTranspose3d
from utils import Constants


class VaeDecoder(VaeComponent):
    def __init__(self, latent_dim, data_dim, latent_private, net_type: NetworkTypes):
        """
        Base for all decoders

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: tuple
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :net_type: network type used as encoder
        :type net_type: NetworkTypes
        """
        super().__init__(latent_dim, data_dim, latent_private, net_type, NetworkRoles.DECODER)


class Dec_CNN(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        CNN decoder for RGB images of size 64x64x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: private latent space size (optional)
        :type latent_private: int
        """
        super(Dec_CNN, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.CNN)

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 512
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        self.n_chan = 3

        # Fully connected lay
        self.lin1 = torch.nn.DataParallel(nn.Linear(self.out_dim, hidden_dim))
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
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
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
        d = torch.sigmoid(x.view(*z.size()[:-1], *self.data_dim))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d.squeeze().reshape(-1, *self.data_dim), torch.tensor(0.75).to(z.device)


class Dec_SVHN(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        Image decoder for the SVHN dataset or images 32x32x3

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_SVHN, self).__init__(latent_dim, data_dim, latent_private,  net_type=NetworkTypes.CNN)
        self.data_dim = data_dim
        self.net_type = "CNN"
        self.linear = nn.Linear(self.out_dim, 128)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, dilation=1)
        self.relu = nn.ReLU()

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        zs = z["latents"]
        bs = zs.shape[:2] if len(zs.squeeze().shape) == 3 else None
        zs = zs.squeeze(0)
        zs = self.linear(zs)
        zs = zs.reshape(-1, zs.size(-1), 1, 1)
        x_hat = self.relu(zs)
        x_hat = self.conv1(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv3(x_hat)
        x_hat = self.relu(x_hat)
        d = torch.sigmoid(self.conv4(x_hat)).permute(0, 2, 3, 1)
        if bs:
            d = d.reshape(*bs, *d.shape[1:])
        return d.squeeze(), torch.tensor(0.75).to(zs.device)


def extra_hidden_layer(hidden_dim):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


class Dec_MNIST2(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, num_hidden_layers=1):
        """
        Decoder for MNIST image data.as originally implemented in https://github.com/iffsid/mmvae

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param num_hidden_layers: how many hidden layers to add
        :type num_hidden_layers: int
        """
        super(Dec_MNIST2, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.FNN)
        modules = []
        self.data_dim = data_dim
        hidden_dim = 400
        self.net_type = "FNN"
        data_d = int(prod(data_dim))
        modules.append(nn.Sequential(nn.Linear(self.out_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dim, data_d)

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.reshape(*z.size()[:-1], *[1, 28, 28]))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale

class Dec_RESCNN(VaeDecoder):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, latent_dim, data_dim, latent_private):
        super(Dec_RESCNN, self).__init__(latent_dim, data_dim, latent_private,  net_type=NetworkTypes.CNN)
        ch = 64
        latent_channels = 512
        channels = 3
        self.conv_t_up = nn.ConvTranspose2d(latent_dim, ch * 16, 4, 1)
        self.res_up_block1 = ResUp(ch * 16, ch * 8)
        self.res_up_block2 = ResUp(ch * 8, ch * 4)
        self.res_up_block3 = ResUp(ch * 4, ch * 2)
        self.res_up_block4 = ResUp(ch * 2, ch)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = x["latents"]
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0]*x.shape[1], -1)
        x = x.squeeze().unsqueeze(-1).unsqueeze(-1)
        try:
            x = self.act_fnc(self.conv_t_up(x))  # 4
        except:
            pass
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = self.res_up_block4(x)  # 64
        x = torch.sigmoid(self.conv_out(x))
        return x, torch.tensor(0.75).to(x.device)


class Dec_MNIST(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        Image decoder for the MNIST images

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_MNIST, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.FNN)
        self.data_dim = data_dim
        self.net_type = "CNN"
        self.hidden_dim = 400
        modules = []
        modules.append(nn.Sequential(nn.Linear(self.out_dim, self.hidden_dim), nn.ReLU(True)))
        modules.extend(
            [nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True)) for _ in range(2 - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        x_hat = self.dec(z)
        x_hat = self.fc3(x_hat)
        x_hat = self.sigmoid(x_hat)
        d = x_hat.reshape(*z.size()[:-1], *self.data_dim).squeeze(0)
        d = d.permute(0, 3, 1, 2) if len(d.shape) == 4 else d.permute(0, 1, 4, 2, 3)
        return d.squeeze(0), torch.tensor(0.75).to(z.device)

class Dec_PolyMNIST(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        Image decoder for the PolyMNIST images from https://github.com/gr8joo/MVTCAE/blob/master/mmnist/networks/ConvNetworksImgCMNIST.py

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_PolyMNIST, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.FNN)
        self.data_dim = data_dim
        self.net_type = "CNN"
        self.hidden_dim = 400
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        x_hat = self.decoder(z.view(-1, self.out_dim))
        x_hat = self.sigmoid(x_hat)
        d = x_hat.view(*z.size()[:-1], *self.data_dim).squeeze(0)
        d = d.permute(0, 3, 1, 2) if len(d.shape) == 4 else d.permute(0, 1, 4, 2, 3)
        return d.squeeze(0), torch.tensor(0.75).to(z.device)

class Dec_SVHN2(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        Decoder for SVHN image data.as originally implemented in https://github.com/iffsid/mmvae

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_SVHN2, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.CNN)
        fBase = 32
        imgChans = 3
        self.net_type = "CNN"
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.out_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(z.device)  # mean, length scale


class Dec_FNN(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        Fully connected layer decoder for any type of data

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data defined in config (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_FNN, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.FNN)
        self.net_type = "FNN"
        self.hidden_dim = 128
        self.data_dim = data_dim
        self.first = torch.nn.DataParallel(nn.Linear(self.out_dim, self.hidden_dim))
        #self.lin2 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        #self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, np.prod(data_dim)))

    def forward(self, z):
        """
        Forward pass

        :param z: sampled latent vectors z
        :type z: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = z["latents"]
        p = F.leaky_relu(self.first(z))
        #p = F.leaky_relu(self.lin2(p))
        #p = F.leaky_relu(self.lin3(p))
        d = (self.fc3(p)) # reshape data
        d = d.reshape(-1, *self.data_dim).squeeze(-1)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class Dec_TransformerIMG(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        """
        Decoder for a sequence of images

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
        super(Dec_TransformerIMG, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.TRANSFORMER)
        self.net_type = "Transformer"
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.data_dim = data_dim
        self.dropout = dropout
        self.activation = activation
        # iteration over image sequence
        self.sequence_pos_encoder = torch.nn.DataParallel(PositionalEncoding(self.self.out_dim, self.dropout))
        seqTransDecoderLayer = torch.nn.DataParallel(nn.TransformerDecoderLayer(d_model=self.self.out_dim,
                                                                                nhead=self.num_heads,
                                                                                dim_feedforward=self.ff_size,
                                                                                dropout=self.dropout,
                                                                                activation=activation))
        self.seqTransDecoder = torch.nn.DataParallel(nn.TransformerDecoder(seqTransDecoderLayer,
                                                                           num_layers=self.num_layers))
        # deconvolution to images
        hid_channels = 64
        kernel_size = 4
        cnn_kwargs = dict(stride=2, padding=1)
        self.reshape = (hid_channels, kernel_size, kernel_size)
        self.lin = torch.nn.DataParallel(nn.Linear(self.self.out_dim, np.product(self.reshape)))
        self.deconvolve = torch.nn.DataParallel(
            torch.nn.Sequential(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                torch.nn.SiLU(),
                                nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                torch.nn.SiLU(),
                                nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                torch.nn.SiLU(),
                                nn.ConvTranspose2d(hid_channels, 3, kernel_size, **cnn_kwargs),
                                torch.nn.Sigmoid()))

    def forward(self, batch):
        """
        Forward pass

        :param batch: list with sampled latent vectors z and (optionally) boolean masks for desired lengths
        :type batch: list, torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = batch["latents"]
        mask = batch["masks"]
        latent_dim = z.shape[-1]
        bs = z.shape[1]
        mask = mask.to(z.device) if mask is not None else torch.tensor(np.ones((bs, self.data_dim[0]), dtype=bool)).to(
            z.device)
        timequeries = torch.zeros(mask.shape[1], bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        images = []
        for i in range(output.shape[0]):
            im = self.lin(output[i])
            images.append(self.deconvolve(im.view(-1, *self.reshape)))
        output = torch.stack(images).permute(1, 0, 3, 4, 2)
        return output.to(z.device), torch.tensor(0.75).to(z.device)


class Dec_VideoGPT(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, n_res_layers=4):
        """
        Decoder for image sequences taken from https://github.com/wilson1yan/VideoGPT

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [10, 64, 64, 3] for 64x64x3 image sequences with max length 10 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        :param n_res_layers: number of ResNet layers
        :type n_res_layers: int
        """
        super(Dec_VideoGPT, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.DCNN)
        self.net_type = "3DCNN"
        self.upsample = (1, 4, 4)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(self.out_dim)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU())
        n_times_upsample = np.array([int(math.log2(d)) for d in self.upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else self.out_dim
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(self.out_dim, out_channels, 4, stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1
        self.upsample = torch.nn.DataParallel(nn.Linear(self.out_dim, self.out_dim * 16 * 16 * self.data_dim[0]))

    def forward(self, z):
        """
        Forward pass

        :param x: sampled latent vectors z
        :type x: torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        x = z["latents"]
        x_upsampled = self.upsample(x)
        h = self.res_stack(x_upsampled.view(-1, x.shape[2], self.data_dim[0], 16, 16))
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        h = h.permute(0, 2, 3, 4, 1)
        h = torch.sigmoid(h)
        return h, torch.tensor(0.75).to(x.device)


class Dec_Transformer(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, ff_size=1024, num_layers=4, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer decoder for arbitrary sequential data

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
        super(Dec_Transformer, self).__init__(latent_dim, data_dim, latent_private, net_type=NetworkTypes.TRANSFORMER)
        self.net_type = "Transformer"
        self.njoints = data_dim[1]
        if len(data_dim) > 2:
            self.nfeats = data_dim[2]
        else:
            self.nfeats = 1
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.sequence_pos_encoder = torch.nn.DataParallel(PositionalEncoding(self.out_dim, self.dropout))

        seqTransDecoderLayer = (nn.TransformerDecoderLayer(d_model=self.out_dim,
                                                                                nhead=self.num_heads,
                                                                                dim_feedforward=self.ff_size,
                                                                                dropout=self.dropout,
                                                                                activation=activation))
        self.seqTransDecoder = (nn.TransformerDecoder(seqTransDecoderLayer,
                                                                           num_layers=self.num_layers))
        self.finallayer = torch.nn.DataParallel(nn.Linear(self.out_dim, self.input_feats))

    def forward(self, batch):
        """
        Forward pass

        :param batch: list with sampled latent vectors z and (optionally) boolean masks for desired lengths
        :type batch: list, torch.tensor
        :return: output reconstructions, log variance
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        z = batch["latents"]
        mask = batch["masks"]
        z = z.reshape(-1, self.out_dim).unsqueeze(0)
        bs = z.shape[1]
        if mask is not None:
            if bs > mask.shape[0]:
                mask = mask.repeat(int(bs / mask.shape[0]), 1)
            mask = mask.to(z.device)
        else:
            mask = torch.tensor(np.ones((bs, self.data_dim[0]), dtype=bool)).to(z.device)
        timequeries = torch.zeros(mask.shape[1], bs, self.out_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(mask.shape[1], bs, self.njoints, self.nfeats)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2, 3)
        return output.to(z.device), torch.tensor(0.75).to(z.device)

class Dec_ConvTxt(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private):
        """
        Decoder configured for text reconstructions

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_ConvTxt, self).__init__(latent_dim, data_dim, latent_private,  net_type=NetworkTypes.TXTTRANSFORMER)
        fBase = 64
        self.data_dim = data_dim
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.out_dim, fBase * 3, 3, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 3),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 3, fBase * 3, (1, 3), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 3),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 3, fBase * 3, (1, 3), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 3),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 3, fBase * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 3, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 3, 2, 1, bias=False),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(1105, self.data_dim[0]*self.data_dim[1])

    def forward(self, z):
        z = z["latents"]
        z = z[0].unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z)
        out = torch.sigmoid(self.toVocabSize(out.view(-1,1105)).view(-1, self.data_dim[0], self.data_dim[1]))
        return out, torch.tensor(0.75).to(z.device)

class Dec_TxtTransformer(VaeDecoder):
    def __init__(self, latent_dim, data_dim, latent_private, ff_size=128, num_layers=1, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer decoder configured for character-level text reconstructions

        :param latent_dim: latent vector dimensionality
        :type latent_dim: int
        :param data_dim: dimensions of the data (e.g. [64,64,3] for 64x64x3 images)
        :type data_dim: list
        :param latent_private: (optional) size of the private latent space in case of latent factorization
        :type latent_private: int
        """
        super(Dec_TxtTransformer, self).__init__(latent_dim, data_dim, latent_private,
                                         net_type=NetworkTypes.TXTTRANSFORMER)
        self.net_type = "Transformer"
        self.njoints = data_dim[1]
        if len(data_dim) > 2:
            self.nfeats = data_dim[2]
        else:
            self.nfeats = 1
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats
        self.sigmoid = nn.Sigmoid()
        seqTransDecoderLayer = (nn.TransformerDecoderLayer(d_model=self.out_dim,
                                                                                nhead=self.num_heads,
                                                                                dim_feedforward=self.ff_size,
                                                                                dropout=self.dropout,
                                                                                activation=activation))
        self.seqTransDecoder = (nn.TransformerDecoder(seqTransDecoderLayer,
                                                                           num_layers=self.num_layers))
        self.finallayer = torch.nn.DataParallel(nn.Linear(self.out_dim, self.input_feats))
        self.sequence_pos_encoder = torch.nn.DataParallel(PositionalEncoding(self.out_dim, self.dropout))

    def forward(self, batch):
        z = batch["latents"]
        z = z.unsqueeze(0) if len(z.shape) == 2 else z
        mask = batch["masks"]
        latent_dim = z.shape[-1]
        bs = z.shape[1]
        mask = mask.to(z.device) if mask is not None else torch.tensor(np.ones((bs, self.data_dim[0]), dtype=bool)).to(
            z.device)
        timequeries = torch.zeros(mask.shape[1], bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = (self.finallayer((output)).reshape(mask.shape[1], bs, self.njoints, self.nfeats)).squeeze(-1)
        # zero for padded area
        output = output.permute(1, 0, 2) * mask.unsqueeze(dim=-1).repeat(1, 1, self.njoints).float()
        return output.to(z.device), torch.tensor(0.75).to(z.device)


