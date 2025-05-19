import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_layer(old_layer, layer_class, new_size):
    """Adds new neurons to a layer, returns expanded layer"""
    weights = old_layer.weight.data
    new_layer = layer_class(in_features=new_size[0], out_features=new_size[1])
    new_layer.weight.data[:weights.shape[0], :weights.shape[1]] = weights
    return new_layer

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class MultiTransformer(nn.Module):
    def __init__(self, latent_dim, zero_masking, decoder, use_ml_layers,  output_mean, pos_encoding=1,ff_size=2048, num_layers=2, num_heads=2, dropout=0.1, activation="gelu"):
        """
        Transformer network for multimodal fusion
        :param latent_dim: int, latent vector dimensionality
        :param data_dim: list, dimensions of the data (e.g. [42, 25, 3] for sequences of max. length 42, 25 joints and 3 features per joint)
        """
        super(MultiTransformer, self).__init__()
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.zero_masking = int(zero_masking) == 1
        self.pos_encoding = pos_encoding == 1 
        self.output_mean = int(output_mean) == 1
        self.use_decoder = decoder==1
        self.use_ml_layers = use_ml_layers==1

        self.input_feats = self.latent_dim 
        self.mu_layer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.latent_dim))
        self.logvar_layer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.latent_dim))

        self.skelEmbedding = torch.nn.DataParallel(nn.Linear(self.input_feats, self.latent_dim))
        if self.pos_encoding:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = (nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                                nhead=self.num_heads,
                                                                                dim_feedforward=self.ff_size,
                                                                                dropout=self.dropout,
                                                                                activation=self.activation))
        self.seqTransEncoder = (
            nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers))
        if self.use_decoder:
            seqTransDecoderLayer = (nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation))
            self.seqTransDecoder = (nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers))


    def forward(self, batch):
        x = batch.float()
        nframes, bs, njoints, nfeats = x.shape
        mask = torch.tensor(np.ones((bs, nframes), dtype=bool)).cuda()
        if self.zero_masking:
            for ix, sample in enumerate(x.permute(1,0,2,3)):
                for ix2, frame in enumerate(sample):
                     if torch.count_nonzero(frame) == 0:
                            mask[ix][ix2] = torch.tensor(False).cuda()
        x = x.reshape(nframes, bs, nfeats*njoints)
        # embedding of the skeleton
        x = self.skelEmbedding(x.cuda())
        # add positional encoding
        if self.pos_encoding:
            x = self.sequence_pos_encoder(x)
        # transformer layers
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        if self.use_decoder:
            timequeries = torch.zeros(mask.shape[1], bs, self.latent_dim, device=final.device)
            timequeries = self.sequence_pos_encoder(timequeries)
            final = self.seqTransDecoder(tgt=timequeries, memory=final, tgt_key_padding_mask=~mask)
        if not self.use_ml_layers:
            mu = final[0]
            logvar = final[1]
        else:
            if self.output_mean:
                z = final.mean(axis=0)
            else:
                z = final[0]
            mu = self.mu_layer(z)
            logvar = self.logvar_layer(z)
        logvar = F.softmax(logvar, dim=-1)
        return mu, logvar

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)

class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


def make_res_block_encoder_feature_compressor(channels_in, channels_out, a_val=2, b_val=0.3):
    downsample = None
    if channels_in != channels_out:
        downsample = nn.Sequential(nn.Conv1d(channels_in,
                                             channels_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             dilation=1),
                                   nn.BatchNorm1d(channels_out))
    layers = [ResidualBlock1dConv(channels_in, channels_out, kernelsize=1, stride=1, padding=0, dilation=1,
                                  downsample=downsample, a=a_val, b=b_val)]
    return nn.Sequential(*layers)


def make_layers_resnet_encoder_feature_compressor(start_channels, end_channels, a=2, b=0.3, l=1):
    layers = []
    num_compr_layers = int((1 / float(l)) * np.floor(np.log(start_channels / float(end_channels))))
    for k in range(0, num_compr_layers):
        in_channels = np.round(start_channels / float(2 ** (l * k))).astype(int)
        out_channels = np.round(start_channels / float(2 ** (l * (k + 1)))).astype(int)
        resblock = make_res_block_encoder_feature_compressor(in_channels, out_channels, a_val=a, b_val=b)
        layers.append(resblock)

    out_channels = np.round(start_channels / float(2 ** (l * num_compr_layers))).astype(int)
    if out_channels > end_channels:
        resblock = make_res_block_encoder_feature_compressor(out_channels, end_channels, a_val=a, b_val=b)
        layers.append(resblock)
    return nn.Sequential(*layers)


class ResidualFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels_style, out_channels_content, a, b, compression_power):
        super(ResidualFeatureCompressor, self).__init__()
        self.a = a
        self.b = b
        self.compression_power = compression_power
        self.style_mu = make_res_block_encoder_feature_compressor(in_channels, out_channels_style, a_val=self.a,
                                                                  b_val=self.b)
        self.style_logvar = make_res_block_encoder_feature_compressor(in_channels, out_channels_style, a_val=self.a,
                                                                      b_val=self.b)
        self.content_mu = make_res_block_encoder_feature_compressor(in_channels, out_channels_content, a_val=self.a,
                                                                    b_val=self.b)
        self.content_logvar = make_res_block_encoder_feature_compressor(in_channels, out_channels_content, a_val=self.a,
                                                                        b_val=self.b)

    def forward(self, feats):
        mu_style, logvar_style = self.style_mu(feats), self.style_logvar(feats)
        mu_content, logvar_content = self.content_mu(feats), self.content_logvar(feats)
        return mu_style, logvar_style, mu_content, logvar_content


def make_res_block_encoder_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation,
                                             a_val=2.0, b_val=0.3):
    downsample = None
    if (stride != 1) or (in_channels != out_channels) or dilation != 1:
        downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation),
                                   nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(
        ResidualBlock1dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample, a=a_val,
                            b=b_val))
    return nn.Sequential(*layers)


class LinearFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels_style, out_channels_content):
        super(LinearFeatureCompressor, self).__init__()
        self.style_mu = nn.Linear(in_channels, out_channels_style, bias=True)
        self.style_logvar = nn.Linear(in_channels, out_channels_style, bias=True)
        self.content_mu = nn.Linear(in_channels, out_channels_content, bias=True)
        self.content_logvar = nn.Linear(in_channels, out_channels_content, bias=True)

    def forward(self, feats):
        feats = feats.view(feats.size(0), -1)
        mu_style, logvar_style = self.style_mu(feats), self.style_logvar(feats)
        mu_content, logvar_content = self.content_mu(feats), self.content_logvar(feats)
        return mu_style, logvar_style, mu_content, logvar_content


class ResidualBlock1dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=2, b=0.3):
        super(ResidualBlock1dConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding,
                               dilation=dilation)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = self.a * residual + self.b * out
        return out


class ResidualBlock1dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, o_padding, upsample, a=2,
                 b=0.3):
        super(ResidualBlock1dTransposeConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride,
                                        padding=padding, dilation=dilation, output_padding=o_padding)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.upsample = upsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.upsample:
            residual = self.upsample(x)
        out = self.a * residual + self.b * out
        return out


def res_block_decoder(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=2.0,
                      b_val=0.3):
    upsample = None

    if (kernelsize != 1 or stride != 1) or (in_channels != out_channels) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(
        ResidualBlock1dTransposeConv(in_channels, out_channels, kernelsize, stride, padding, dilation, o_padding,
                                     upsample=upsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


class DataGeneratorText(nn.Module):
    def __init__(self, data_dim, a, b, DIM_text=128):
        super(DataGeneratorText, self).__init__()
        self.datadim = data_dim
        self.DIM_text = DIM_text
        self.a = a
        self.b = b
        self.resblock_1 = res_block_decoder(5 * self.DIM_text, 5 * self.DIM_text,
                                            kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0)
        self.resblock_2 = res_block_decoder(5 * self.DIM_text, 5 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_3 = res_block_decoder(5 * self.DIM_text, 4 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_4 = res_block_decoder(4 * self.DIM_text, 3 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_5 = res_block_decoder(3 * self.DIM_text, 2 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_6 = res_block_decoder(2 * self.DIM_text, self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.conv2 = nn.ConvTranspose1d(self.DIM_text, self.datadim[1],
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=0)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feats):
        d = self.resblock_1(feats)
        d = self.resblock_2(d)
        d = self.resblock_3(d)
        d = self.resblock_4(d)
        d = self.resblock_5(d)
        d = self.resblock_6(d)
        d = self.conv2(d)
        d = self.softmax(d)
        return d


class FeatureExtractorText(nn.Module):
    def __init__(self, datadim, a, b, dim_text=128):
        super(FeatureExtractorText, self).__init__()
        self.txtdim = datadim
        self.a = a
        self.b = b
        self.DIM_text = dim_text
        self.conv1 = nn.Conv1d(self.txtdim[1], self.DIM_text,
                               kernel_size=4, stride=2, padding=1, dilation=1)
        self.resblock_1 = make_res_block_encoder_feature_extractor(self.DIM_text, 2 * self.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_2 = make_res_block_encoder_feature_extractor(2 * self.DIM_text, 3 * self.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_3 = make_res_block_encoder_feature_extractor(3 * self.DIM_text, 4 * self.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_4 = make_res_block_encoder_feature_extractor(4 * self.DIM_text, 5 * self.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_5 = make_res_block_encoder_feature_extractor(5 * self.DIM_text, 5 * self.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_6 = make_res_block_encoder_feature_extractor(5 * self.DIM_text, 5 * self.DIM_text,
                                                                   kernelsize=4, stride=2, padding=0, dilation=1)

    def forward(self, x):
        x = x.transpose(-2, -1)
        out = self.conv1(x)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        try:
            x = x + self.pe[:x.shape[0], :]
        except:
            x = x.permute(1, 0, 2, 3) + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class DeconvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(DeconvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ResidualBlockDeconv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=dilation_size,  # (kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(ConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=dilation_size,  # (kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False)  # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False)  # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False)  # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True)  # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), **attn_kwargs)
        elif attn_type == 'sparse':
            self.attn = SparseAttention(shape, n_head, causal, **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None
        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                      v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i + 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a)  # (b x seq_len x embd_dim)

        return a


class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)

        return view_range(out, 2, 3, old_shape)


class SparseAttention(nn.Module):
    ops = dict()
    attn_mask = dict()
    block_layout = dict()

    def __init__(self, shape, n_head, causal, num_local_blocks=4, block=32,
                 attn_dropout=0.):  # does not use attn_dropout
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.sparsity_config = StridedSparsityConfig(shape=shape, n_head=n_head,
                                                     causal=causal, block=block,
                                                     num_local_blocks=num_local_blocks)

        if self.shape not in SparseAttention.block_layout:
            SparseAttention.block_layout[self.shape] = self.sparsity_config.make_layout()
        if causal and self.shape not in SparseAttention.attn_mask:
            SparseAttention.attn_mask[self.shape] = self.sparsity_config.make_sparse_attn_mask()

    def get_ops(self):
        try:
            from deepspeed.ops.sparse_attention import MatMul, Softmax
        except:
            raise Exception(
                'Error importing deepspeed. Please install using `DS_BUILD_SPARSE_ATTN=1 pip install deepspeed`')
        if self.shape not in SparseAttention.ops:
            sparsity_layout = self.sparsity_config.make_layout()
            sparse_dot_sdd_nt = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

            SparseAttention.ops[self.shape] = (sparse_dot_sdd_nt,
                                               sparse_dot_dsd_nn,
                                               sparse_softmax)
        return SparseAttention.ops[self.shape]

    def forward(self, q, k, v, decode_step, decode_idx):
        if self.training and self.shape not in SparseAttention.ops:
            self.get_ops()

        SparseAttention.block_layout[self.shape] = SparseAttention.block_layout[self.shape].to(q)
        if self.causal:
            SparseAttention.attn_mask[self.shape] = SparseAttention.attn_mask[self.shape].to(q).type_as(q)
        attn_mask = SparseAttention.attn_mask[self.shape] if self.causal else None

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        if decode_step is not None:
            mask = self.sparsity_config.get_non_block_layout_row(SparseAttention.block_layout[self.shape], decode_step)
            out = scaled_dot_product_attention(q, k, v, mask=mask, training=self.training)
        else:
            if q.shape != k.shape or k.shape != v.shape:
                raise Exception('SparseAttention only support self-attention')
            sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops()
            scaling = float(q.shape[-1]) ** -0.5

            attn_output_weights = sparse_dot_sdd_nt(q, k)
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask == 0,
                                                                      float('-inf'))
            attn_output_weights = sparse_softmax(
                attn_output_weights,
                scale=scaling
            )

            out = sparse_dot_dsd_nn(attn_output_weights, v)

        return view_range(out, 2, 3, old_shape)


def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


class StridedSparsityConfig(object):
    """
    Strided Sparse configuration specified in https://arxiv.org/abs/1904.10509 that
    generalizes to arbitrary dimensions
    """

    def __init__(self, shape, n_head, causal, block, num_local_blocks):
        self.n_head = n_head
        self.shape = shape
        self.causal = causal
        self.block = block
        self.num_local_blocks = num_local_blocks

        assert self.num_local_blocks >= 1, 'Must have at least 1 local block'
        assert self.seq_len % self.block == 0, 'seq len must be divisible by block size'

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    @property
    def seq_len(self):
        return np.prod(self.shape)

    @property
    def num_blocks(self):
        return self.seq_len // self.block

    def set_local_layout(self, layout):
        num_blocks = self.num_blocks
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                    max(0, row - self.num_local_blocks),
                    (row + 1 if self.causal else end)):
                layout[:, row, col] = 1
        return layout

    def set_global_layout(self, layout):
        num_blocks = self.num_blocks
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[:, row, col] = 1

        return layout

    def make_layout(self):
        layout = torch.zeros((self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64)
        layout = self.set_local_layout(layout)
        layout = self.set_global_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        block_layout = self.make_layout()
        assert block_layout.shape[1] == block_layout.shape[2] == self.num_blocks

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(self.n_head):
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j:  # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]]  # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1:] = 0.
        return block_row

    ############# Helper functions ##########################

    def _compute_block_shape(self):
        n_dim = len(self.shape)
        cum_prod = 1
        for i in range(n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)

        assert np.prod(new_shape) == np.prod(self.shape) // self.block

        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(self._block_shape), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)


class DataGeneratorText(nn.Module):
    def __init__(self, data_dim, a, b, dim_text=128):
        super(DataGeneratorText, self).__init__()
        self.data_dim = data_dim
        self.DIM_text = dim_text
        self.a = a
        self.b = b
        self.resblock_1 = res_block_decoder(5 * self.DIM_text, 5 * self.DIM_text,
                                            kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0)
        self.resblock_2 = res_block_decoder(5 * self.DIM_text, 5 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_3 = res_block_decoder(5 * self.DIM_text, 4 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_4 = res_block_decoder(4 * self.DIM_text, 3 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_5 = res_block_decoder(3 * self.DIM_text, 2 * self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.resblock_6 = res_block_decoder(2 * self.DIM_text, self.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0)
        self.conv2 = nn.ConvTranspose1d(self.DIM_text, self.data_dim[1],
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=0)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feats):
        d = self.resblock_1(feats)
        d = self.resblock_2(d)
        d = self.resblock_3(d)
        d = self.resblock_4(d)
        d = self.resblock_5(d)
        d = self.resblock_6(d)
        d = self.conv2(d)
        d = self.softmax(d)
        return d


def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn
    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)
    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x d
    return a


class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2  # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.af = nn.Tanh()
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.net = nn.Sequential(
            self.conv1, self.af, nn.Dropout(dropout), self.conv2, self.af, nn.Dropout(dropout)
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.af(out + res)


class ResidualBlockDeconv(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(ResidualBlockDeconv, self).__init__()
        self.conv1 = nn.ConvTranspose1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.af = nn.Tanh()
        self.conv2 = nn.ConvTranspose1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.net = nn.Sequential(
            self.conv1, self.af, nn.Dropout(dropout), self.conv2, self.af, nn.Dropout(dropout)
        )
        self.upsample = (
            nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x):
        out = self.net(x)
        res = x if self.upsample is None else self.upsample(x)
        return self.af(out + res)

class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()

    def load_params_(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x):
        return (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)

        return loss/16