import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import Constants, create_vocab, W2V
from models.nn_modules import PositionalEncoding, ConvNet, ResidualBlock

def unpack(d):
    if isinstance(d, list):
        while len(d) == 1:
            d = d[0]
        d = torch.tensor(d)
    return d

# Classes
class Enc_CNN(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, data_dim):
        super(Enc_CNN, self).__init__()
        self.name = "CNN"
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
        self.pooling = torch.nn.AvgPool2d(kernel_size)
        # Fully connected layers
        self.lin1 = torch.nn.DataParallel(nn.Linear(np.product(self.reshape), hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(hidden_dim, hidden_dim))

        # Fully connected layers for mean and variance
        self.mu_gen = torch.nn.DataParallel(nn.Linear(hidden_dim, self.latent_dim))
        self.var_gen = torch.nn.DataParallel(nn.Linear(hidden_dim, self.latent_dim))

    def forward(self, x):
        batch_size = x.size(0) if len(x.shape) == 4 else x.size(1)
        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x.float()))
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

# Classes
class Enc_FNN(nn.Module):
    def __init__(self, latent_dim, data_dim=1):
        super(Enc_FNN, self).__init__()
        self.name = "FNN"
        self.hidden_dim = 300
        self.lin1 = torch.nn.DataParallel(nn.Linear(np.prod(data_dim), self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(np.prod(data_dim), self.hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.fc21 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, latent_dim))
        self.fc22 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, latent_dim))

    def forward(self, x):
        x = (x).float()
        e = torch.relu(self.lin1(x.view(x.shape[0], -1)))
        e = torch.relu(self.lin2(e))
        e = torch.relu(self.lin3(e))
        lv = self.fc22(e)
        lv =  F.softmax(lv, dim=-1) + Constants.eta
        return self.fc21(e), lv


class Enc_Audio(nn.Module):
    def __init__(self, latent_dim, data_dim=1):
        super(Enc_Audio, self).__init__()
        self.name = "AudioConv"
        self.latent_dim = latent_dim
        self.TCN = ConvNet(data_dim[0], [128, 128, 96, 96, 64], dropout=0)
        self.share_mean = nn.Sequential(nn.Linear(64*data_dim[-1], 32), nn.ReLU(), nn.Linear(32, self.latent_dim))
        self.share_logvar = nn.Sequential(nn.Linear(64*data_dim[-1], 32), nn.ReLU(), nn.Linear(32, self.latent_dim))

    def forward(self, inputs):
        """Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs.float()).permute(0,2,1)
        x = output.reshape(inputs.shape[0], -1)
        mu = self.share_mean(x)
        logvar = self.share_logvar(x)
        return mu, logvar


class Enc_Transformer(nn.Module):
    """ Transformer VAE as implemented in https://github.com/Mathux/ACTOR"""
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu"):
        super(Enc_Transformer, self).__init__()
        self.name = "Transformer"
        self.njoints = data_dim[1]
        self.nfeats = data_dim[2]
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats
        self.mu_layer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_layer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.latent_dim))

        self.skelEmbedding = torch.nn.DataParallel(nn.Linear(self.input_feats, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))

    def forward(self, batch):
        x, mask = (batch[0]).float(), batch[1]
        bs, nframes, njoints, nfeats = x.shape
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
        logvar = self.sigma_layer(z)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar
