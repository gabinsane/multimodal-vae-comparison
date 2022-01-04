import torch, numpy as np
import torch.nn as nn
from utils import Constants, create_vocab, W2V


# Constants
dataSize = torch.Size([3,64,64])

class Dec_CNN(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, data_dim):
        super(Dec_CNN, self).__init__()
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

class Dec_FCNN(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim, data_dim=1):
        super(Dec_FCNN, self).__init__()
        self.hidden_dim = 20
        self.data_dim = data_dim
        self.lin1 = torch.nn.DataParallel(nn.Linear(latent_dim, self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(latent_dim, self.hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, np.prod(data_dim)))

    def forward(self, z):
        p = torch.relu(self.lin1(z))
        #p = torch.relu(self.lin2(p))
        #p = torch.relu(self.lin3(p))
        d = (self.fc3(p))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale

