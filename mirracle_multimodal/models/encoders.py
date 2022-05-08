import torch, numpy as np
import torch.nn as nn
import torchvision as visnn
import chainer.links as L
import chainer, math
from chainer import cuda, Variable
import torch.nn.functional as F
from utils import Constants, create_vocab, W2V
from models.nn_modules import PositionalEncoding, ConvNet, ResidualBlock, SamePadConv3d, AttentionResidualBlock, FeatureExtractorText, LinearFeatureCompressor
import torchvision.transforms as transforms

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
        self.mu_layer = torch.nn.DataParallel(nn.Linear(hidden_dim, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(nn.Linear(hidden_dim, self.latent_dim))

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
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

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
        return self.fc21(e), lv


class Enc_Audio(nn.Module):
    def __init__(self, latent_dim, data_dim=1):
        super(Enc_Audio, self).__init__()
        self.name = "AudioConv"
        self.latent_dim = latent_dim
        self.TCN = ConvNet(data_dim[0], [128, 128, 96, 96, 64], dropout=0)
        self.mu_layer = nn.Sequential(nn.Linear(64*data_dim[-1], 32), nn.ReLU(), nn.Linear(32, self.latent_dim))
        self.logvar_layer = nn.Sequential(nn.Linear(64*data_dim[-1], 32), nn.ReLU(), nn.Linear(32, self.latent_dim))

    def forward(self, inputs):
        """Args:
            inputs: input tensor of shape: (B, T, C)
        """
        inputs = torch.stack(inputs).cuda() if isinstance(inputs, list) else inputs
        output = self.TCN(inputs.float()).permute(0,2,1)
        x = output.reshape(inputs.shape[0], -1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        #logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar

class Enc_TransformerIMG(nn.Module):
    """ Transformer VAE as implemented in https://github.com/Mathux/ACTOR"""
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu"):
        super(Enc_TransformerIMG, self).__init__()
        self.name = "Transformer"
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.datadim = data_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        #self.conv_pretrained = visnn.models.resnet152(pretrained=True, progress=True)
        hid_channels = 32
        kernel_size = 4
        n_chan = 3
        self.reshape = (hid_channels, kernel_size, kernel_size)
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.convolve = torch.nn.DataParallel(torch.nn.Sequential(nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU(),
                                                nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU(),
                                                nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU(),
                                                nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU()))
        self.downsample = torch.nn.DataParallel(nn.Linear(np.product(self.reshape), self.latent_dim))
        #Transformer layers
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        self.mu_layer = torch.nn.DataParallel(nn.Linear(self.datadim[0] * self.latent_dim, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(nn.Linear(self.datadim[0] *  self.latent_dim, self.latent_dim))

    def forward(self, batch):
        if isinstance(batch, list):
            x, mask = batch
            x = torch.stack(x).cuda() if isinstance(x, list) else x
        else:
            x, mask = batch, None
        bs, nframes = x.shape[0], x.shape[1]
        imgs_feats = []
        for i in range(x.shape[1]):
            #imgs_feats.append(self.downsample(self.conv_pretrained(x[:, i, :].permute(0, 3, 2, 1).float())))
            imgs_feats.append(self.downsample(self.convolve(x[:, i, :].permute(0,3,2,1).float()).view(-1, np.prod(self.reshape))))
        x = torch.stack(imgs_feats)
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = self.sequence_pos_encoder(x)
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        # extract mu and logvar
        mu = self.mu_layer(final.view(bs, -1))
        logvar = self.logvar_layer(final.view(bs, -1))
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_Text(nn.Module):
    def __init__(self, latent_dim, data_dim=1, dim_text = 64):
        super(Enc_Text, self).__init__()
        self.name = "textonehot"
        self.DIM_text = dim_text
        self.feature_extractor = FeatureExtractorText(data_dim, a=2.0, b=0.3)
        self.feature_compressor = LinearFeatureCompressor(5*self.DIM_text, 32, latent_dim)

    def forward(self, x_text):
        h_text = self.feature_extractor(x_text)
        mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_text);
        return mu_style, logvar_style, mu_content, logvar_content, h_text;


class Enc_Conv2(nn.Module):
    def __init__(self, latent_dim, channel=1, density=1, initial_size=64, Normal=1):
        super(Enc_Conv2, self).__init__()
        self.dc1= L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                            initialW=Normal(0.02))
        self.dc2= L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                            initialW=Normal(0.02))
        self.norm2= L.BatchNormalization(int(32 * density))
        self.dc3= L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                            initialW=Normal(0.02))
        self.norm3= L.BatchNormalization(int(64 * density))
        self.dc4= L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                            initialW=Normal(0.02))
        self.norm4= L.BatchNormalization(int(128 * density))
        self.mean= L.Linear(initial_size * initial_size * int(128 * density), latent_dim,
                      initialW=Normal(0.02))
        self.var= L.Linear(initial_size * initial_size * int(128 * density), latent_dim,
                     initialW=Normal(0.02))

    def forward(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            mean = self.mean(h4)
            var = self.var(h4)
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.clip(F.exp(var), .001, 100.) * Variable(rand)
            # z  = mean + F.exp(var) * Variable(rand, volatile=not train)
            return mean, var


class Enc_VideoGPT(nn.Module):
    def __init__(self,  latent_dim, data_dim=1, n_res_layers=4, downsample=(2,4,4)):
        super(Enc_VideoGPT, self).__init__()
        self.name = "3DCNN"
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else latent_dim
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, latent_dim, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, latent_dim, kernel_size=3)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(latent_dim)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(latent_dim),
            nn.ReLU())
        self.mu_layer = torch.nn.DataParallel(nn.Linear(latent_dim*16*16, latent_dim))
        self.logvar_layer = torch.nn.DataParallel(nn.Linear(latent_dim*16*16, latent_dim))

    def forward(self, x):
        h = x.permute(0,4,1,2,3)
        for conv in self.convs:
            h = F.relu(conv(h.float()))
        h = self.conv_last(h)
        h = self.res_stack(h)
        mu = self.mu_layer(h.view(x.shape[0], -1))
        logvar = self.logvar_layer(h.view(x.shape[0], -1))
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar


class Enc_Transformer(nn.Module):
    """ Transformer VAE as implemented in https://github.com/Mathux/ACTOR"""
    def __init__(self, latent_dim, data_dim, ff_size=1024, num_layers=8, num_heads=2, dropout=0.1, activation="gelu"):
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
        self.logvar_layer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.latent_dim))

        self.skelEmbedding = torch.nn.DataParallel(nn.Linear(self.input_feats, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))

    def forward(self, batch):
        if isinstance(batch[0], list):
            x = torch.stack(batch[0]).float()
        else:
            x = (batch[0]).float()
        mask = batch[1]
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

class Enc_TxtTransformer(Enc_Transformer):
    def __init__(self, latent_dim, data_dim=1):
        super(Enc_TxtTransformer, self).__init__(latent_dim=latent_dim, data_dim=data_dim)
        self.name = "TxtTransformer"
        self.embedding = nn.Embedding(self.input_feats,2)
        self.sequence_pos_encoder = PositionalEncoding(2, self.dropout)
        seqTransEncoderLayer = torch.nn.DataParallel(nn.TransformerEncoderLayer(d_model=self.input_feats*2,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation))
        self.seqTransEncoder = torch.nn.DataParallel(nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        self.mu_layer = torch.nn.DataParallel(nn.Linear(self.input_feats*2, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(nn.Linear(self.input_feats*2, self.latent_dim))

    def forward(self, batch):
        if isinstance(batch[0], list):
            x = torch.stack(batch[0]).float()
        else:
            x = (batch[0]).float()
        mask = batch[1]
        bs, nframes, njoints = x.shape
        mask = mask if mask is not None else torch.tensor(np.ones((bs, x.shape[1]), dtype=bool)).cuda()
        x = self.embedding(x.cuda().long())
        x = self.sequence_pos_encoder(x)
        final = self.seqTransEncoder(x.view(nframes, bs, -1), src_key_padding_mask=~mask)
        z = final.mean(axis=0)
        # extract mu and logvar
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        logvar = F.softmax(logvar, dim=-1) + Constants.eta
        return mu, logvar