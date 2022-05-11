import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.nn_modules import PositionalEncoding, DeconvNet
from utils import Constants, create_vocab, W2V
import chainer, math
#import chainer.functions as F
import chainer.links as L
from chainer.initializers import Normal
from models.nn_modules import PositionalEncoding, ConvNet, ResidualBlock, SamePadConv3d, AttentionResidualBlock,\
    SamePadConvTranspose3d, DataGeneratorText

class Dec_CNN(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, latent_dim, data_dim):
        super(Dec_CNN, self).__init__()
        latent_dim = latent_dim
        self.datadim = data_dim
        self.name = "CNN"

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
        d = torch.sigmoid(x.view(*z.size()[:-1], *self.datadim))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d.squeeze().reshape(-1, *self.datadim), torch.tensor(0.75).to(z.device)

class Dec_FNN(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim, data_dim=1):
        super(Dec_FNN, self).__init__()
        self.name = "FNN"
        self.hidden_dim = 30
        self.data_dim = data_dim
        self.lin1 = torch.nn.DataParallel(nn.Linear(latent_dim, self.hidden_dim))
        self.lin2 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lin3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = torch.nn.DataParallel(nn.Linear(self.hidden_dim, np.prod(data_dim)))

    def forward(self, z):
        p = torch.relu(self.lin1(z))
        p = torch.relu(self.lin2(p))
        p = torch.relu(self.lin3(p))
        d = (self.fc3(p))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale

class Dec_Audio(nn.Module):
    def __init__(self, latent_dim, data_dim=1):
        super(Dec_Audio, self).__init__()
        self.name = "AudioConv"
        self.latent_dim = latent_dim
        self.reshape = (64, 3)
        self.data_dim = data_dim
        self.lin1 = torch.nn.DataParallel(nn.Linear(latent_dim, np.product(self.reshape)))
        self.TCN = DeconvNet(self.reshape[0], [64,96,96,128,128], dropout=0)
        self.output_layer = nn.Sequential(nn.Linear(128*3, np.prod(data_dim)))

    def forward(self, z):
        """Args: z: input tensor of shape: (B, T, C)
        """
        out = torch.relu(self.lin1(z))
        output = self.TCN(out.float().reshape(-1, *self.reshape))
        x = output.reshape(-1, 128*3)
        output = self.output_layer(x)
        return output.reshape(-1, *self.data_dim), torch.tensor(0.75).to(z.device)

class Dec_TransformerIMG(nn.Module):
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        super(Dec_TransformerIMG, self).__init__()
        self.name = "Transformer"
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.data_dim = data_dim
        self.dropout = dropout
        self.activation = activation
        # iteration over image sequence
        self.sequence_pos_encoder = torch.nn.DataParallel(PositionalEncoding(self.latent_dim, self.dropout))
        seqTransDecoderLayer = torch.nn.DataParallel(nn.TransformerDecoderLayer(d_model=self.latent_dim,
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
        self.lin = torch.nn.DataParallel(nn.Linear(self.latent_dim, np.product(self.reshape)))
        self.deconvolve = torch.nn.DataParallel(torch.nn.Sequential(nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU(),
                                                nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU(),
                                                nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs),
                                                torch.nn.SiLU(),
                                                nn.ConvTranspose2d(hid_channels, 3,  kernel_size, **cnn_kwargs),
                                                torch.nn.Sigmoid()))
    def forward(self, batch):
        if isinstance(batch, list):
            z, mask = batch
        else:
            z = batch
            mask = None
        latent_dim = z.shape[-1]
        bs = z.shape[1]
        mask = mask.to(z.device) if mask is not None else torch.tensor(np.ones((bs, self.data_dim[0]), dtype=bool)).to(z.device)
        timequeries = torch.zeros(mask.shape[1], bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        images = []
        for i in range(output.shape[0]):
            im = self.lin(output[i])
            images.append(self.deconvolve(im.view(-1, *self.reshape)))
        output = torch.stack(images).permute(1,0,3,4,2)
        return output.to(z.device), torch.tensor(0.75).to(z.device)


class Decoder_Conv2(nn.Module):
    def __init__(self, latent_dim, data_dim=1, density=1, initial_size=64, channel=3):
        super(Decoder_Conv2, self).__init__()
        self.g1 = L.Linear(latent_dim, initial_size * initial_size * 128 * density, initialW=Normal(0.02))
        self.norm1 = L.BatchNormalization(initial_size * initial_size * 128 * density)
        self.g2 = L.Deconvolution2D(128 * density, 64 * density, 4, stride=2, pad=1,
                             initialW=Normal(0.02))
        self.norm2 = L.BatchNormalization(64 * density)
        self.g3 = L.Deconvolution2D(64 * density, 32 * density, 4, stride=2, pad=1,
                             initialW=Normal(0.02))
        self.norm3 = L.BatchNormalization(32 * density)
        self.g4 = L.Deconvolution2D(32 * density, 16 * density, 4, stride=2, pad=1,
                             initialW=Normal(0.02))
        self.norm4 = L.BatchNormalization(16 * density)
        self.g5 = L.Deconvolution2D(16 * density, channel, 4, stride=2, pad=1,
                             initialW=Normal(0.02))
        self.g2_= L.Deconvolution2D(64 * density, 64 * density, 3, stride=1, pad=1,
                              initialW=Normal(0.02))
        self.norm2_= L.BatchNormalization(64 * density)
        self.g3_= L.Deconvolution2D(32 * density, 32 * density, 3, stride=1, pad=1,
                              initialW=Normal(0.02))
        self.norm3_= L.BatchNormalization(32 * density),
        self.g4_= L.Deconvolution2D(16 * density, 16 * density, 3, stride=1, pad=1,
                              initialW=Normal(0.02))
        self.norm4_= L.BatchNormalization(16 * density)
        self.g5_ = L.Deconvolution2D(channel, channel, 3, stride=1, pad=1, initialW=Normal(0.02))
        self.density = density
        self.latent_size = latent_dim
        self.initial_size = initial_size

    def forward(self, z, train=True):
        with chainer.using_config('train', train):
            h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)),
                           (z.data.shape[0], 128 * self.density, self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h2_ = F.relu(self.norm2_(self.g2_(h2)))
            h3 = F.relu(self.norm3(self.g3(h2_)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h4 = F.relu(self.norm4(self.g4(h3_)))
            h4_ = F.relu(self.norm4_(self.g4_(h4)))
            return F.tanh(self.g5(h4_)), torch.tensor(0.75).to(z.device)


class Dec_VideoGPT(nn.Module):
    def __init__(self, latent_dim, data_dim=1, n_res_layers=4, upsample=(1,4,4)):
        super(Dec_VideoGPT, self).__init__()
        self.name = "3DCNN"
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(latent_dim)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(latent_dim),
            nn.ReLU())
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else latent_dim
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(latent_dim, out_channels, 4, stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1
        self.upsample = torch.nn.DataParallel(nn.Linear(latent_dim, latent_dim*16*16*3))

    def forward(self, x):
        x_upsampled = self.upsample(x)
        h = self.res_stack(x_upsampled.view(-1, x.shape[2], 3, 16, 16))
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        h = h.permute(0, 1, 3, 4, 2)
        return h, torch.tensor(0.75).to(x.device)


class Dec_Text(nn.Module):
    def __init__(self, latent_dim, data_dim=1, dim_text=128):
        super(Dec_Text, self).__init__()
        self.name = "textonehot"
        self.DIM_text = dim_text
        self.feature_generator = nn.Linear(latent_dim, 5*self.DIM_text, bias=True);
        self.text_generator = DataGeneratorText(data_dim, a=2.0, b=0.3)

    def forward(self, z):
        z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
        text_feat_hat = self.feature_generator(z);
        text_feat_hat = text_feat_hat.unsqueeze(-1);
        text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2,-1);
        return text_hat



class Dec_Transformer(nn.Module):
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=4, num_heads=2, dropout=0.1, activation="gelu"):
        super(Dec_Transformer, self).__init__()
        self.name = "Transformer"
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
        self.sequence_pos_encoder = torch.nn.DataParallel(PositionalEncoding(self.latent_dim, self.dropout))

        seqTransDecoderLayer = torch.nn.DataParallel(nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation))
        self.seqTransDecoder = torch.nn.DataParallel(nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers))
        self.finallayer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, batch):
        z, mask = batch[0], batch[1]
        latent_dim = z.shape[-1]
        bs = z.shape[1]
        mask = mask.to(z.device) if mask is not None else torch.tensor(np.ones((bs, self.data_dim[0]), dtype=bool)).to(z.device)
        timequeries = torch.zeros(mask.shape[1], bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(mask.shape[1], bs, self.njoints,  self.nfeats)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2, 3)
        return output.to(z.device), torch.tensor(0.75).to(z.device)

class Dec_TxtTransformer(Dec_Transformer):
    def __init__(self, latent_dim, data_dim=1):
        super(Dec_TxtTransformer, self).__init__(latent_dim, data_dim, ff_size=1024, num_layers=2, num_heads=4,)
        self.name = "TxtTransformer"
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.ConvTranspose1d(self.latent_dim, self.input_feats,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=0)

    def forward(self, batch):
        z, mask = batch[0], batch[1]
        latent_dim = z.shape[-1]
        bs = z.shape[1]
        mask = mask.to(z.device) if mask is not None else torch.tensor(np.ones((bs, self.data_dim[0]), dtype=bool)).to(z.device)
        timequeries = torch.zeros(mask.shape[1], bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = (self.finallayer(self.sigmoid(output)).reshape(mask.shape[1], bs, self.njoints,  self.nfeats)).squeeze(-1)
        # zero for padded area
        output = output.permute(1,0,2) * mask.unsqueeze(dim=-1).repeat(1,1,self.njoints).float()
        #output = self.softmax(output)
        #one_pos = torch.argmax(output, dim=-1)
        #output = torch.zeros_like(output).scatter_(1, one_pos.unsqueeze(-1), 1.).squeeze()
        return output.to(z.device), torch.tensor(0.75).to(z.device)
