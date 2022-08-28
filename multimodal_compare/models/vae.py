# Base VAE class definition
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from models import encoders, decoders
from models.decoders import VaeDecoder
from models.encoders import VaeEncoder
from utils import get_mean, kl_divergence, load_images, lengths_to_mask

class DencoderFactory(object):
    @classmethod
    def get_nework_classes(cls, enc_name, dec_name, n_latents, data_dim:tuple):
        """
        Instantiates the encoder and decoder networks

       :param enc: encoder name
       :type enc: str
       :param dec: decoder name
       :type dec: str
       :return: returns encoder and decoder class
       :rtype: tuple(object, object)
       """
        assert hasattr(encoders, "Enc_{}".format(enc_name)), "Did not find encoder {}".format(enc_name)
        enc_obj = getattr(encoders, "Enc_{}".format(enc_name))(n_latents, data_dim)
        assert hasattr(decoders, "Dec_{}".format(dec_name)), "Did not find decoder {}".format(dec_name)
        dec_obj = getattr(decoders, "Dec_{}".format(dec_name))(n_latents, data_dim)
        return enc_obj, dec_obj


class BaseVae(nn.Module):
    def __init__(self, enc, dec, prior_dist=dist.Normal, likelihood_dist=dist.Normal, post_dist=dist.Normal):
        super().__init__()
        assert isinstance(enc, VaeEncoder) and isinstance(dec, VaeDecoder)
        self.device = None
        self.enc = enc
        self.dec = dec
        assert enc.latent_dim == dec.latent_dim
        assert enc.modality_type == dec.modality_type
        self.n_latents = enc.latent_dim
        self.modality_type = enc.modality_type
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist

    def forward(self, x, K=1):
        """
        Forward pass

        :param x: input modality
        :type x: torch.tensor
        :param K: sample K samples from the posterior
        :type K: int
        :return: the posterior distribution, the reconstruction and latent samples
        :rtype:tuple(torch.dist, torch.dist, torch.tensor)
        """
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        if "transformer" in self.dec.net_type.lower():
            px_z = self.px_z(*self.dec([zs, x[1]] if x is not None else [zs, None]))
        else:
            px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs


class VAE(BaseVae):
    def __init__(self, enc, dec, feature_dim, n_latents, prior_dist=dist.Normal,
                 likelihood_dist=dist.Normal, post_dist=dist.Normal):
        """
        The general unimodal VAE module, can be used separately or in a multimodal VAE

        :param enc: encoder name
        :type enc: str
        :param dec: decoder name
        :type dec: str
        :param feature_dim: data dimensionality as stated in config
        :type feature_dim: list
        :param n_latents: latent space dimensionality
        :type n_latents: int
        :param prior_dist: prior distribution
        :type prior_dist: torch.dist
        :param likelihood_dist: likelihood distribution
        :type likelihood_dist: torch.dist
        :param post_dist: posterior distribution
        :type post_dist: torch.dist
        """
        enc_net, dec_net = DencoderFactory().get_nework_classes(enc, dec, n_latents, feature_dim)
        super(VAE, self).__init__(enc_net, dec_net, prior_dist, likelihood_dist, post_dist)
        self._qz_x_params = None
        self.llik_scaling = 1.0
        self.data_dim = feature_dim
        self.n_latents = n_latents
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False)  # logvar
        ])
        self.modelName = 'vae_{}'.format(enc)

    @property
    def pz_params(self):
        """
        :return: returns likelihood parameters
        :rtype: list(torch.tensor, torch.tensor)
        """
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @property
    def qz_x_params(self):
        """
        :return: returns posterior distribution parameters
        :rtype: list(torch.tensor, torch.tensor)
        """
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    def forward(self, x, K=1):
        """
        Forward pass
        :param x: input modality
        :type x: torch.tensor
        :param K: sample K samples from the posterior
        :type K: int
        :return: the posterior distribution, the reconstruction and latent samples
        :rtype:tuple(torch.dist, torch.dist, torch.tensor)
        """
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec({"latents":zs, "masks": None}))
        return qz_x, px_z, zs

    def generate_samples(self, N):
        """
        Generates samples from the latent space
        :param N: How many samples to make
        :type N: int
        :return: output reconstructions
        :rtype: torch.tensor
        """
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            if "transformer" in self.enc_name.lower():
                px_z = self.px_z(*self.dec([latents, None]))
            else:
                px_z = self.px_z(*self.dec(latents))
            data = get_mean(px_z)
        return data
