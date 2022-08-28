# Base MMVAE class definition, common for PoE, MoE, MoPoE
import abc
from itertools import combinations
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from utils import get_mean, kl_divergence, lengths_to_mask, output_onehot2text
from data_proc.process_audio import numpy_to_wav
from visualization import t_sne, tensors_to_df, plot_kls_df
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import numpy as np
from torch.autograd import Variable
import cv2, math
from .vae import VAE


class BaseMMVAE(object):
    modelname = 'basemmvae'

    @abc.abstractmethod
    def forward(self, inputs, K=1):
        """
        Forward pass that takes input data and outputs a list of private and shared posteriors, reconstructions and latent samples

        :param inputs: input data, a list of modalities where missing modalities are replaced with None
        :type inputs: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: a list of posterior distributions, a list of reconstructions and latent samples
        :rtype: tuple(list, list, list)
        """
        pass

    @abc.abstractmethod
    def infer(self, inputs):
        """
        Inference module, calculates the joint posterior

        :param inputs: list of input modalities, missing mods are replaced with None
        :type inputs: list
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        pass

    @abc.abstractmethod
    def reconstruct(self, data):
        """
        Reconstructs the input data

        :param data: list of input modalities
        :type data: list
        :return: reconstructions
        :rtype: list
        """
        pass


class TorchMMVAE(BaseMMVAE, nn.Module):
    """
    Base class for all PyTorch based MMVAE implementations.
    """

    def __init__(self):
        super().__init__()

        self.vaes = []
        self.mod_types = []
        self.data_dims = []

    def forward(self, inputs, K=1, mods=[]):
        pass

    def encode(self, inputs, K=1, mods=[]):
        pass

    def decode(self, x, K=1, mods=[]):
        pass

    def infer(self, inputs):
        """
        Inference module, calculates the joint posterior

        :param inputs: list of input modalities, missing mods are replaced with None
        :type inputs: list
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        id = 0 if inputs[0] is not None else 1
        batch_size = len(inputs[id]) if len(inputs[id]) != 2 else len(inputs[id][0])
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents), use_cuda=True)
        for ix, modality in enumerate(inputs):
            if modality is not None:
                mod_mu, mod_logvar = self.vaes[ix].enc(
                    modality.to("cuda") if not isinstance(modality, list) else modality)
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)
        mu_before, logvar_before = mu, logvar
        # product of experts to combine gaussians
        mu, logvar = self.product_of_experts(mu, logvar)
        return mu, logvar, [mu_before[1:], logvar_before[1:]]

    def reconstruct(self, data):
        """
        Reconstructs the input data

        :param data: list of input modalities
        :type data: list
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
