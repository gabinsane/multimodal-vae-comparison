# Multi-modal model specification
from models.mmvae_base import TorchMMVAE
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from utils import combinatorial, Constants
from torch.autograd import Variable
from models.NetworkTypes import VaeOutput

class MOE(TorchMMVAE):
    def __init__(self, vaes, model_config=None):
        """
        Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae
        :param vaes: list of modality-specific vae objects
        :type encoders: list
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__()
        self.model_config = model_config
        self.vaes = nn.ModuleDict(vaes)
        self.modelName = 'moe'
        self.prior_dist = dist.Normal
        self.pz = dist.Normal

    def pz_params(self):
        """
        :return: returns parameters of the prior distribution
        :rtype: tuple(nn.Parameter, nn.Parameter)
        """
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

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
            if val is None:
                keys.append(modality)
            else:
                keys_with_val.append(modality)
        return keys, keys_with_val

    def forward(self, x, K=1):
        """
        Forward pass that takes input data and outputs a list of posteriors, reconstructions and latent samples
        :param x: input data, a list of modalities where missing modalities are replaced with None
        :type x: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: a list of posterior distributions, a list of reconstructions and latent samples
        :rtype: tuple(list, list, list)
        """
        qz_xs = self.encode(x)
        zs = {}
        for modality, qz_x in qz_xs.items():
            qz_xs[modality] = self.vaes[modality].qz_x(*qz_x)
            z = self.vaes[modality].qz_x(*qz_x).rsample(torch.Size([K]))
            zs[modality] = {"latents":z, "masks":None}
        # decode the samples
        px_zs = self.decode(zs)
        for modality, px_z in px_zs.items():
            px_zs[modality] = [dist.Normal(*px_z[0])]
        missing, filled = self.get_missing_modalities(qz_xs)
        assert len(filled) > 1, "at least one modality must be present for forward call"
        for mod_name in missing:
            px_zs[mod_name] = [dist.Normal(*self.vaes[mod_name].dec(zs[filled[0]]))]
        for modality, z in zs.items():
            for mod_vae, vae in self.vaes.items():
                if mod_vae != modality:  # fill-in off-diagonal
                    px_zs[mod_vae].append(vae.px_z(*vae.dec(z)))
        output_dict = {}
        for modality in self.vaes.keys():
            output_dict[modality] = VaeOutput(encoder_dists=qz_xs[modality], decoder_dists=px_zs[modality],
                                                latent_samples=zs[modality])
        return output_dict

    def reconstruct(self, data, runPath, epoch, N=8):
        """
        Reconstruct data for individual experts
        :param data: list of input modalities
        :type data: list
        :param runPath: path to save data to
        :type runPath: str
        :param epoch: current epoch to name the data
        :type epoch: str
        :param N: how many samples to reconstruct
        :type N: int
        """
        recons_mat = super(MOE, self).reconstruct([d for d in data])
        self.process_reconstructions(recons_mat, data, epoch, runPath)


class POE(TorchMMVAE):
    def __init__(self, vaes, model_config=None):
        """
        Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public
        :param vaes: list of modality-specific vae objects
        :type encoders: list
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__()
        self.vaes = nn.ModuleDict(vaes)
        self.model_config = model_config
        self.modelName = 'poe'
        self.pz = dist.Normal
        self.prior_dist = dist.Normal

    def pz_params(self):
        """
        :return: returns parameters of the prior distribution
        :rtype: tuple(nn.Parameter, nn.Parameter)
        """
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def reparameterize(self, mu, logvar):
        """
        General reparametrization trick during training
        :param mu: vector of means
        :type mu: torch.tensor
        :param logvar: vector of log variances
        :type logvar: torch.tensor
        :return: reparametrized samples
        :rtype: torch.tensor
        """

        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, inputs, K=1):
        """
        Forward pass that takes input data and outputs a list of posteriors, reconstructions and latent samples
        :param inputs: input data, a list of modalities where missing modalities are replaced with None
        :type inputs: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: a list of posterior distributions, a list of reconstructions and latent samples
        :rtype: tuple(list, list, list)
        """
        mu, logvar, single_params = self.infer(inputs, K)
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        qz_d, px_d, z_d = {}, {}, {}
        z_d["joint"] = {"latents": z, "masks": None}
        for mod, vae in self.vaes.items():
            px_d[mod] = vae.px_z(*vae.dec(z_d["joint"]))
        output_dict = {}
        qz_d["joint"] = qz_x
        for modality in self.vaes.keys():
            output_dict[modality] = VaeOutput(encoder_dists=qz_d["joint"], decoder_dists=[px_d[modality]],
                                                latent_samples=z_d["joint"])
        return output_dict

    def infer(self,x, K=1):
        """
        Inference module, calculates the joint posterior
        :param x: list of input modalities, missing mods are replaced with None
        :type x: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        for key in x.keys():
            if x[key]["data"] is not None:
                batch_size = x[key]["data"].shape[0]
                break
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.vaes["mod_1"].n_latents), use_cuda=True)
        for m, vae in self.vaes.items():
            if x[m]["data"] is not None:
                mod_mu, mod_logvar = vae.enc(x[m])
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)
        mu_before, logvar_before = mu, logvar
        # product of experts to combine gaussians
        mu, logvar = self.product_of_experts(mu, logvar)
        return mu, logvar, [mu_before[1:], logvar_before[1:]]

    def product_of_experts(self, mu, logvar):
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

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if use_cuda:
            mu, logvar = mu.to("cuda"), logvar.to("cuda")
        return mu, logvar

    def reconstruct(self, data, runPath, epoch, N=64):
        recons_mat = []
        for ix, i in enumerate(data):
            input_mat = [None] * len(data)
            input_mat[ix] = i[:N]
            rec = super(POE, self).reconstruct(input_mat)
            recons_mat.append(rec)
        self.process_reconstructions(recons_mat, data, epoch, runPath)

class MoPOE(TorchMMVAE):
    def __init__(self, vaes, model_config=None):
        """
        Multimodal Variational Autoencoder with Generalized Multimodal Elbo https://github.com/thomassutter/MoPoE
        :param vaes: list of modality-specific vae objects
        :type encoders: list
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__()
        self.vaes = nn.ModuleDict(vaes)
        self.model_config = model_config
        self.modelName = 'mopoe'
        self.pz = dist.Normal
        self.subsets = [[x] for x in self.vaes] + list(combinatorial([x for x in self.vaes]))
        self.prior_dist = dist.Normal

    def pz_params(self):
        """
        :return: returns parameters of the prior distribution
        :rtype: tuple(nn.Parameter, nn.Parameter)
        """
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def forward(self, inputs, K=1):
        """
        Forward pass that takes input data and outputs a list of posteriors, reconstructions and latent samples
        :param inputs: input data, a list of modalities where missing modalities are replaced with None
        :type inputs: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: a list of posterior distributions, a list of reconstructions and latent samples
        :rtype: tuple(list, list, list)
        """
        mu, logvar, single_params = self.infer(inputs)
        recons = []
        qz_x = dist.Normal(*[mu, logvar])
        z = self.reparameterize(mu, logvar).unsqueeze(0)
        for ind, vae in enumerate(self.vaes):
            recons.append(vae.px_z(*vae.dec({"latents": z, "masks": None})))
        return qz_x, recons, [z], single_params
        return qz_x, recons, [z]

    def infer(self, x, num_samples=None):
        mu, logvar = [None] * len(self.vaes), [None] * len(self.vaes)
        for m, vae in enumerate(self.vaes):
            tag = "mod_{}".format(m+1)
            if x[tag]["data"] is not None:
                mod_mu, mod_logvar = vae.enc(x[tag])
                mu[m] = mod_mu.unsqueeze(0)
                logvar[m] = mod_logvar.unsqueeze(0)
        mus, logvars = torch.Tensor().cuda(), torch.Tensor().cuda()
        distr_subsets = dict()
        for k, subset in enumerate(self.subsets):
            mus_subset = torch.Tensor().cuda()
            logvars_subset = torch.Tensor().cuda()
            for vae in subset:
                mod_index = list(self.vaes).index(vae)
                if mu[mod_index] is not None:
                    mus_subset = torch.cat((mus_subset, mu[mod_index].unsqueeze(0)), dim=0)
                    logvars_subset = torch.cat((logvars_subset, logvar[mod_index].unsqueeze(0)), dim=0)
            if mus_subset.nelement() != 0:
                s_mu, s_logvar = self.poe_fusion(mus_subset, logvars_subset)
                distr_subsets[k] = [s_mu, s_logvar]
                if len(mus.shape) > 3:
                    mus = mus.squeeze(0)
                mus = torch.cat((mus, s_mu), dim=0)
                logvars = torch.cat((logvars, s_logvar), dim=0)
        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).cuda()
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)
        return joint_mu, joint_logvar, [mus, logvars]

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def product_of_experts(self, mu, logvar):
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

    def reweight_weights(self, w):
        return w / w.sum()

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = self.reweight_weights(weights)
        mu_moe, logvar_moe = self.mixture_component_selection(mus, logvars, weights)
        return [mu_moe, logvar_moe]

    def poe_fusion(self, mus, logvars):
        if mus.shape[0] == len(self.vaes):
            mus = torch.cat((mus.squeeze(1), torch.zeros(1, mus.shape[-2], self.vaes[0].n_latents).cuda()), dim=0)
            logvars = torch.cat((logvars.squeeze(1), torch.zeros(1, mus.shape[-2], self.vaes[0].n_latents).cuda()), dim=0)
        mu_poe, logvar_poe = self.product_of_experts(mus, logvars)
        if len(mu_poe.shape) < 3:
            mu_poe = mu_poe.unsqueeze(0)
            logvar_poe = logvar_poe.unsqueeze(0)
        return [mu_poe, logvar_poe]

    def mixture_component_selection(sellf, mus, logvars, w_modalities=None):
        num_components, num_samples = mus.shape[0], mus.shape[1]
        idx_start, idx_end = [], []
        for k in range(0, num_components):
            i_start = 0 if k == 0 else int(idx_end[k - 1])
            if k == w_modalities.shape[0] - 1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples
        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
        return [mu_sel, logvar_sel]


class DMVAE(TorchMMVAE):
    def __init__(self, vaes, model_config=None):
        """
        Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations https://github.com/seqam-lab/DMVAE
        :param vaes: list of modality-specific vae objects
        :type encoders: list
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__()
        self.model_config = model_config
        self.vaes = nn.ModuleDict(vaes)
        self.modelName = 'dmvae'
        self.pz = dist.Normal
        self.qz_x = dist.Normal


    def pz_params(self):
        """
        :return: returns parameters of the prior distribution
        :rtype: tuple(nn.Parameter, nn.Parameter)
        """
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def forward(self, x, K=1):
        """
        Forward pass that takes input data and outputs a list of private and shared posteriors, reconstructions and latent samples
        :param inputs: input data, a list of modalities where missing modalities are replaced with None
        :type inputs: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: a list of posterior distributions, a list of reconstructions and latent samples
        :rtype: tuple(list, list, list)
        """
        qz_xs_shared, px_zs= [], [[None for _ in range(len(self.vaes)+1)] for _ in range(len(self.vaes))]
        qz_xs_private = [None for _ in range(len(self.vaes))]
        # initialise cross-modal matrix
        for m, vae in enumerate(self.vaes):
            if x[m] is not None:
                mod_mu, mod_std = self.vaes[m].enc(x[m].to("cuda") if not isinstance(x[m], list) else x[m])
                qz_xs_private[m] = self.vaes[m].qz_x(*[mod_mu[0], mod_std[0]])
                qz_xs_shared.append(self.vaes[m].qz_x(*[mod_mu[1], mod_std[1]]))
        mu_joint, std_joint = self.apply_poe(qz_xs_shared)
        joint_d = self.qz_x(*[mu_joint, std_joint])
        all_shared = qz_xs_shared + [joint_d]
        zss = []
        for d, vae in enumerate(self.vaes):
            for e, dist in enumerate(all_shared):
                    zs_shared = dist.rsample(torch.Size([K]))
                    zs_private = qz_xs_private[d].rsample(torch.Size([K]))
                    zs = torch.cat([zs_private, zs_shared], -1)[0]
                    zss.append(zs)
                    px_zs[d][e] = vae.px_z(*vae.dec({"latents": zs, "masks": None}))
        return qz_xs_private + all_shared, px_zs, zss


    def apply_poe(self, qz_xs_shared):
        """
        Applies the product of experts to the shared posteriors
        :param qz_xs_shared: list of posteriors for all modalities
        :type qz_xs_shared: list
        :return: joint means and standard deviations
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        zero = torch.zeros(qz_xs_shared[0].scale.shape).cuda()
        logvars = [-torch.log(x.scale ** 2 + Constants.eps) for x in qz_xs_shared]
        logvarShared = -self.logsumexp(torch.stack((zero, *logvars), dim=2), dim=2)
        stdS = torch.sqrt(torch.exp(logvarShared))

        muS = 0
        for dist in qz_xs_shared:
            muS += dist.loc / (dist.scale ** 2 + Constants.eps)
        muS = muS * (stdS ** 2)
        return muS, stdS


    def logsumexp(self, x, dim=None, keepdim=False):
        """
        A smooth maximum function
        :param x: input data
        :type x: torch.tensor
        :param dim: dimension
        :type dim: int
        :param keepdim: whether to keep shape or squeeze
        :type keepdim: bool
        :return: data
        :rtype: torch.tensor
        """
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            (xm == float('inf')) | (xm == float('-inf')),
            xm,
            xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
        return x if keepdim else x.squeeze(dim)
