# Base MMVAE class definition, common for PoE, MoE, MoPoE
import abc
import torch, os
import torch.nn as nn
from models.output_storage import VAEOutput
import torch.distributions as dist
from models.objectives import MultimodalObjective
from models.vae import BaseVae


class TorchMMVAE(nn.Module):
    """
    Base class for all PyTorch based MMVAE implementations.
    """

    def __init__(self, n_latents: int, obj: str, beta=1):
        """
        :param n_latents: dimensionality of the (shared) latent space
        :type n_latents: int
        :param obj: name of the objective (elbo/iwae)
        :type obj: str
        :param beta: beta parameter for weighted KL Divergence
        :type beta: float
        """
        super().__init__()
        self.vaes = nn.ModuleDict()
        self.modelName = 'TorchMMVAE'
        self.qz_x = dist.Normal
        self.px_z = dist.Normal
        self.pz = dist.Normal
        self.n_latents = n_latents
        self.obj_fn = MultimodalObjective(obj, beta)

    @property
    def pz_params(self):
        return nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, self.n_latents), requires_grad=False)  # logvar
        ])

    @property
    def latent_factorization(self):
        """Returns True if latent space is factorized into shared and modality-specific subspaces, else False"""
        for vae in self.vaes.keys():
            if self.vaes[vae].private_latents is not None:
                return True
        return False


    def add_vaes(self, vae_dict: nn.ModuleDict):
        """
        This functions updates the VAEs of the MMVAE with a given dictionary.
        Args:
            vae_dict: A dictionary with the modality names as keys and BaseVAEs as values
            type vae_dict: nn.ModuleDict
        """
        if not all(isinstance(key, str) for key in vae_dict.keys()):
            raise ValueError(f'Expected modality name as str, but got {list(vae_dict.keys())}.')
        if not all(isinstance(vae, BaseVae) for vae in vae_dict.values()):
            raise ValueError(f'Expected modality name as str, but got {list(vae_dict.values())}.')
        self.vaes.update(vae_dict)
        print(f'Updated MMVae has the following modalities: {list(self.vaes.keys())}')

    def forward(self, inputs, K=1):
        """
        The general forward pass of multimodal VAE

        :param inputs: input dictionary with modalities as keys and data tensors as values
        :type inputs: dict
        :param K: number of samples
        :type K: int
        :return: dictionary with modalities as keys and namedtuples as values
        :rtype: dict[str,VaeOutput]
        """

        # encode all present inputs using corresponding VAEs
        qz_xs = self.encode(inputs)
        qz_xs = self.modality_mixing(qz_xs)

        # sample from each distribution
        zs = {}
        for modality, qz_x in qz_xs.items():
            qz_xs[modality] = self.vaes[modality].qz_x(*qz_x["shared"])
            z = self.vaes[modality].qz_x(*qz_x["shared"]).rsample(torch.Size([K]))
            zs[modality] = {"latents": z, "masks": None}

        # decode the samples
        px_zs = self.decode(zs)
        for modality, px_z in px_zs.items():
            px_zs[modality] = dist.Normal(*px_z)
        return self.make_output_dict(qz_xs, px_zs, zs)

    def make_output_dict(self, encoder_dist=None, decoder_dist=None, latent_samples=None,
                         joint_dist=None, enc_dist_private=None, dec_dist_private=None,
                         joint_decoder_dist=None, cross_decoder_dist=None):
        """
        Prepares output of the forward pass

        :param encoder_dist: dict with modalities as keys and encoder distributions as values
        :type encoder_dist: dict
        :param decoder_dist: dict with modalities as keys and decoder distributions as values
        :type decoder_dist: dict
        :param latent_samples: dict with modalities as keys and dicts with latent samples as values
        :type latent_samples: dict
        :param joint_dist: dict with modalities as keys and joint distribution as values
        :type joint_dist: dict
        :param enc_dist_private: dict with modalities as keys and dicts with single latent distributions as values
        :type enc_dist_private: dict
        :param dec_dist_private: dict with modalities as keys and dicts with single decoder distributions as values
        :type dec_dist_private: dict
        :param joint_decoder_dist: dict with modalities as keys and dicts with decoder distributions coming from joint distribution
        :type joint_decoder_dist: dict
        :param cross_decoder_dist: dict with modalities as keys and dicts with cross-modal decoder distributions
        :type cross_decoder_dist: dict
        :return: VAEOutput object
        :rtype: object
        """
        out = VAEOutput()
        for v in ["encoder_dist", "decoder_dist", "latent_samples", "joint_dist", "enc_dist_private", "dec_dist_private",
                  "joint_decoder_dist", "cross_decoder_dist"]:
            out.set_with_dict(locals()[v], v)
        return out

    def encode(self, inputs):
        """
        Encode inputs with appropriate VAE encoders

        :param inputs: input dictionary with modalities as keys and data tensors as values
        :type inputs: dict
        :return: qz_xs dictionary with modalities as keys and distribution parameters as values
        :rtype: dict
        """
        qz_xs = {}
        for modality, vae in self.vaes.items():
            if modality in inputs and not inputs[modality]["data"] is None:
                qz_x = vae.enc(inputs[modality])
                if not self.latent_factorization:
                    qz_xs[modality] = {"shared": qz_x, "private": None}
                else:
                    qz_xs[modality] = {"shared":[qz_x[0][:,:vae.n_latents], qz_x[1][:,:vae.n_latents]],
                                       "private": [qz_x[0][:,vae.n_latents:], qz_x[1][:,vae.n_latents:]]}
            elif modality in inputs and inputs[modality]["data"] is None:
                qz_xs[modality] = {"shared": None, "private": None}
        return qz_xs

    @abc.abstractmethod
    def modality_mixing(self, mods):
        """
        Mix the encoded distributions according to the chosen approach

        :param mods: qz_xs dictionary with modalities as keys and distribution parameters as values
        :type mods: dict
        :return: latent samples dictionary with modalities as keys and latent sample tensors as values
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def objective(self, mods):
        """
        Includes the forward pass and calculates the loss

        :param mods: dictionary with input data with modalities as keys
        :type mods: dict
        :return: loss
        :rtype: dict
        """
        pass

    def decode(self, samples):
        """
        Make reconstructions for the input samples

        :param samples: Dictionary with modalities as keys and latent sample tensors as values
        :type samples: dict
        :return: dictionary with modalities as keys and torch distributions as values
        :rtype: dict
        """
        pz_xs = {}
        for modality, vae in self.vaes.items():
            if modality in samples and not samples[modality]["latents"] is None:
                pz_x = vae.dec(samples[modality])
                pz_xs[modality] = pz_x
            elif modality in samples and samples[modality]["latents"] is None:
                pz_xs[modality] = [None]
        return pz_xs

    @staticmethod
    def product_of_experts(mu, logvar):
        """
        Calculated the product of experts for input data

        :param mu: list of means
        :type mu: list
        :param logvar: list of logvars
        :type logvar: list
        :return: joint posterior
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        eps = 1e-8
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = pd_var
        return pd_mu, pd_logvar

    def get_missing_modalities(self, mods):
        """
        Get indices of modalities that are missing

        :param mods: list of modalities
        :type mods: list
        :return: list of indices of missing modalities
        :rtype: list
        """
        keys = []
        keys_with_val = []
        for modality, val in mods.items():
            if val["data"] is None:
                keys.append(modality)
            else:
                keys_with_val.append(modality)
        return keys, keys_with_val
