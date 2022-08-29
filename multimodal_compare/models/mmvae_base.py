# Base MMVAE class definition, common for PoE, MoE, MoPoE
import abc
import torch, os
import torch.nn as nn
from models.NetworkTypes import VaeOutput
from utils import get_mean, kl_divergence, lengths_to_mask, output_onehot2text
import torch.distributions as dist

class TorchMMVAE(nn.Module):
    """
    Base class for all PyTorch based MMVAE implementations.
    """

    def __init__(self):
        super().__init__()
        self.vaes = {}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.vaes[0].n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, self.vaes[0].n_latents), requires_grad=False)  # logvar
        ])

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
            qz_x = self.qz_x(*self._qz_x_params)
            qz_x = dist.Normal(*[qz_x])
            z = qz_x.rsample(torch.Size([K]))
            zs[modality] = z

        # decode the samples
        px_zs = self.decode(zs)
        output_dict = {}
        for modality in self.vaes.keys():
            output_dict[modality] = VaeOutput(encoder_dists=qz_xs[modality], decoder_dists=px_zs[modality],
                                                latent_samples=zs[modality])
        return output_dict


    def encode(self, inputs, K=1):
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
                qz_x = vae.enc(inputs[modality], K=K)
                qz_xs[modality] = qz_x
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

    def decode(self, samples, K=1):
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
            if modality in samples and not samples[modality] is None:
                pz_x = vae.dec(samples[modality], K=K)
                pz_xs[modality] = pz_x
        return pz_xs


    def infer(self, inputs):
        """
        Inference module, calculates the joint posterior

        :param inputs: list of input modalities, missing mods are replaced with None
        :type inputs: list
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        raise NotImplementedError

    def reconstruct(self, data):
        """
        Reconstructs the input data

        :param data: dict of input modalities
        :type data: dict
        :return: reconstructions
        :rtype: list
        """
        self.eval()
        with torch.no_grad():
            res = self.forward(data)
            px_zs = res[1]
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs] if any(isinstance(i, list) for i in px_zs) \
                else [get_mean(px_z) for px_z in px_zs]
        return recons
