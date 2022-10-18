# Base VAE class definition
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from models import encoders, decoders
from models.decoders import VaeDecoder
from models.encoders import VaeEncoder
from models.NetworkTypes import VaeOutput
from models.objectives import UnimodalObjective

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
    """
    Base VAE class for all implementations.
    """
    def __init__(self, enc, dec, prior_dist=dist.Normal, likelihood_dist=dist.Normal, post_dist=dist.Normal):
        """
        :param enc: encoder class instance
        :type enc: VaeEncoder
        :param dec: decoder class instance
        :type dec: VaeDecoder
        :param prior_dist: prior distribution
        :type prior_dist: torch.distributions
        :param likelihood_dist: likelihood distribution
        :type likelihood_dist: torch.distributions
        :param post_dist: posterior distribution
        :type post_dist: torch.distributions
        """
        super().__init__()
        assert isinstance(enc, VaeEncoder) and isinstance(dec, VaeDecoder)
        self.device = None
        self.enc = enc
        self.dec = dec
        assert enc.latent_dim == dec.latent_dim
        self.n_latents = enc.latent_dim
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist

    def encode(self, inp):
        """
        Encodes the inputs

        :param inp: Inputs dictionary
        :type inp: dict
        :return: encoded distribution parameters (means and logvars)
        :rtype: tuple
        """
        return self.enc(inp)

    def decode(self, inp):
        """
        Decodes the latent samples

        :param inp: Samples dictionary
        :type inp: dict
        :return: decoded distribution parameters (means and logvars)
        :rtype: tuple
        """
        return self.dec(inp)

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
        self._qz_x_params = self.encode(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z_params = self.decode({"latents":zs, "masks": None})
        px_z = self.px_z(*px_z_params)
        output_dict = {}
        output_dict["mod_1"] = VaeOutput(encoder_dists=qz_x, decoder_dists=px_z,
                                          latent_samples={"latents":zs})
        return output_dict

class VAE(BaseVae):
    def __init__(self, enc, dec, feature_dim, n_latents, ltype, prior_dist=dist.Normal,
                 likelihood_dist=dist.Normal, post_dist=dist.Normal, obj_fn=None, beta=1, id_name="mod_1"):
        """
        The general unimodal VAE module, can be used separately or in a multimodal VAE

        :param enc: encoder name
        :type enc: str
        :param dec: decoder name
        :type dec: str
        :param feature_dim: data dimensionality as stated in dataset class
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
        self.post_dist = post_dist
        self.likelihood_dist = likelihood_dist
        self.prior_dist = prior_dist
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, n_latents), requires_grad=False)  # logvar
        ])
        self.modelName = id_name
        self.ltype = ltype
        self.obj_fn = self.set_objective_fn(obj_fn, beta)


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

    def set_objective_fn(self, obj_fn, beta):
        """Set up loss function in case of unimodal VAE"""
        if obj_fn is not None:
            obj = UnimodalObjective(obj_fn, beta)
            obj.set_ltype(self.ltype)
            return obj
        return None

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
        return latents

    def objective(self, data):
        """
        Objective function for unimodal VAE scenario (not used with multimodal VAEs)

        :param data: input data with modalities as keys
        :type data: dict
        :return: loss calculated using self.loss_fn
        :rtype: torch.tensor
        """
        assert self.obj_fn is not None, "loss_fn not defined!"
        output = self.forward(data["mod_1"])
        qz_x = output["mod_1"].encoder_dists
        px_z = output["mod_1"].decoder_dists
        zs = output["mod_1"].latent_samples
        loss = self.obj_fn.calculate_loss(px_z, data["mod_1"]["data"], qz_x, self.prior_dist, self._pz_params, zs)
        return loss
