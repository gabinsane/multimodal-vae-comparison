from ..mmvae_base import TorchMMVAE
import torch
from torch.autograd import Variable

class POE(TorchMMVAE):
    modelname = 'poe'
    def __init__(self, vaes, model_config=None):
        """
        Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae
        :param vaes: list of modality-specific vae objects
        :type encoders: list
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__()
        self.vaes = torch.nn.ModuleDict(vaes)
        self.model_config = model_config
        self.modelName = 'poe'

    def get_batch_size(self, inp):
        """Estimates batch size from the input"""
        batch_size = None
        for key in inp.keys():
            if inp[key] is not None:
                batch_size = inp[key][0].shape[0]
                break
        return batch_size

    def modality_mixing(self, mods):
        """ Calculate the product of experts to get the joint posterior """
        batch_size = self.get_batch_size(mods)
        mu, logvar = self.prior_expert((1, batch_size, self.vaes["mod_1"].n_latents))
        for m, params in mods.items():
            if params is not None:
                mu = torch.cat((mu, params[0].unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, params[1].unsqueeze(0)), dim=0)
        eps = 1e-8
        var = torch.exp(logvar) + eps
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        for key in mods.keys():
            mods[key] = [pd_mu, pd_var]
        return mods

    def prior_expert(self, size):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        """
        mu = Variable(torch.zeros(size)).to("cuda")
        logvar = Variable(torch.log(torch.ones(size))).to("cuda")
        return mu, logvar
