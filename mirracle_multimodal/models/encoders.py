import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import Constants, create_vocab, W2V

# Classes
class Enc_CNN(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, data_dim):
        super(Enc_CNN, self).__init__()
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

# Classes
class Enc_FCNN(nn.Module):
    def __init__(self, latent_dim, data_dim=1):
        super(Enc_FCNN, self).__init__()
        self.hidden_dim = 300
        self.lin1 = torch.nn.DataParallel(nn.Linear(np.prod(data_dim), self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(np.prod(data_dim), self.hidden_dim))
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
