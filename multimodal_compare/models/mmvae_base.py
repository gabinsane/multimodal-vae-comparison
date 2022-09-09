# Base MMVAE class definition, common for PoE, MoE, MoPoE
import abc
import torch, os
import torch.nn as nn
from models.NetworkTypes import VaeOutput
import torch.distributions as dist

from models.vae import BaseVae


class TorchMMVAE(nn.Module):
    """
    Base class for all PyTorch based MMVAE implementations.
    """

    def __init__(self):
        super().__init__()
        self.vaes = nn.ModuleDict()

        self.modelName = 'TorchMMVAE'
        self.prior_dist = dist.Normal
        self.pz = dist.Normal

    def pz_params(self, modality=None):
        if not modality:
            modality = "mod_1"
        return nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.vaes[modality].n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, self.vaes[modality].n_latents), requires_grad=False)  # logvar
        ])

    def add_vaes(self, vae_dict: nn.ModuleDict):
        """
        This functions updates the VAEs of the MMVAE with a given dictionary.
        Args:
            vae_dict: A dictionary with the modality names as keys and BaseVAEs as values
            type vae_dict: nn.ModuleDict

        Returns:

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
            qz_xs[modality] = self.vaes[modality].qz_x(*qz_x)
            z = self.vaes[modality].qz_x(*qz_x).rsample(torch.Size([K]))
            zs[modality] = {"latents": z, "masks": None}

        # decode the samples
        px_zs = self.decode(zs)
        for modality, px_z in px_zs.items():
            px_zs[modality] = [dist.Normal(*p) for p in px_z]
        output_dict = {}
        for modality in self.vaes.keys():
            output_dict[modality] = VaeOutput(encoder_dists=qz_xs[modality], decoder_dists=px_zs[modality],
                                              latent_samples=zs[modality])
        return output_dict

    def encode(self, inputs):
        """
        Encode inputs with appropriate VAE encoders

        :param inputs: input dictionary with modalities as keys and data tensors as values
        :type inputs: dict
        :param K: number of samples
        :type K: int
        :return: qz_xs dictionary with modalities as keys and distribution parameters as values
        :rtype: dict
        """
        qz_xs = {}
        for modality, vae in self.vaes.items():
            if modality in inputs and not inputs[modality]["data"] is None:
                qz_x = vae.enc(inputs[modality])
                qz_xs[modality] = qz_x
            elif modality in inputs and inputs[modality]["data"] is None:
                qz_xs[modality] = None
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

    def decode(self, samples):
        """
        Make reconstructions for the input samples

        :param samples: Dictionary with modalities as keys and latent sample tensors as values
        :type samples: dict
        :param K: number of samples
        :type K: int
        :return: dictionary with modalities as keys and torch distributions as values
        :rtype: dict
        """
        pz_xs = {}
        for modality, vae in self.vaes.items():
            if modality in samples and not samples[modality]["latents"] is None:
                pz_x = vae.dec(samples[modality])
                pz_xs[modality] = [pz_x]
            elif modality in samples and samples[modality]["latents"] is None:
                pz_xs[modality] = [None]
        return pz_xs
