# Multi-modal model specification
import torch
import torch.distributions as dist
from utils import combinatorial
from .mmvae_base import MMVAE
from torch.autograd import Variable
from utils import Constants


class MOE(MMVAE):
    def __init__(self, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size):
        """
        Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae

        :param encoders: list of encoder names (strings) as listed in config
        :type encoders: list
        :param decoders: list of decoder names (strings) as listed in config
        :type decoders: list
        :param data_paths: list of data paths for all modalities
        :type data_paths: list
        :param feature_dims: list of modality-specific feature dimensions as listed in config
        :type feature_dims: list
        :param mod_types: list of modality types (strings) from config
        :type mod_types: list
        :param n_latents: list of latent dimensionalities from config
        :type n_latents: list
        :param test_split: fraction of the data to be used for validation
        :type test_split: float
        :param batch_size: batch size
        :type batch_size: int
        """
        self.modelName = 'moe'
        super(MOE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size)

    def get_missing_modalities(self, mods):
        """
        Get indices of modalities that are missing

        :param mods: list of modalities
        :type mods: list
        :return: list of indices of missing modalities
        :rtype: list
        """
        indices = []
        for i, e in enumerate(mods):
            if e is None:
                indices.append(i)
        return indices

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
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            if x[m] is not None:
                qz_x, px_z, zs = vae(x[m], K=K)
                qz_xs.append(qz_x)
                zss.append(zs)
                px_zs[m][m] = px_z  # fill-in diagonal
        for ind in self.get_missing_modalities(x):
            lat = zss[0] if "transformer" not in self.vaes[ind].dec_name.lower() else [zss[0], None]
            px_zs[ind][ind] = self.vaes[ind].px_z(*self.vaes[ind].dec(lat))
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if "transformer" in self.vaes[d].dec_name.lower():
                        px_zs[e][d] = vae.px_z(*vae.dec([zs, x[d][1]] if x[d] is not None else [zs, None] ))
                    else:
                        px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

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


class POE(MMVAE):
    def __init__(self, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size):
        """
        Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public

        :param encoders: list of encoder names (strings) as listed in config
        :type encoders: list
        :param decoders: list of decoder names (strings) as listed in config
        :type decoders: list
        :param data_paths: list of data paths for all modalities
        :type data_paths: list
        :param feature_dims: list of modality-specific feature dimensions as listed in config
        :type feature_dims: list
        :param mod_types: list of modality types (strings) from config
        :type mod_types: list
        :param n_latents: list of latent dimensionalities from config
        :type n_latents: list
        :param test_split: fraction of the data to be used for validation
        :type test_split: float
        :param batch_size: batch size
        :type batch_size: int
        """
        self.modelName = 'poe'
        super(POE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size)
        self.n_latents = n_latents

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
        if self.modelName == "mopoe":
            z = self.reparameterize(mu, logvar).unsqueeze(0)
        else:
            z = qz_x.rsample(torch.Size([1]))
        for ind, vae in enumerate(self.vaes):
            if "transformer" in vae.dec_name.lower():
               z_dec = [z, inputs[ind][1]] if inputs[ind] is not None else [z, None]
            else: z_dec = z
            recons.append(vae.px_z(*vae.dec(z_dec)))
        if self.modelName == "mopoe":
            return qz_x, recons, [z], single_params
        return qz_x, recons, [z]

    def infer(self,inputs):
        """
        Inference module, calculates the joint posterior

        :param inputs: list of input modalities, missing mods are replaced with None
        :type inputs: list
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        id = 0 if inputs[0] is not None else 1
        batch_size = len(inputs[id]) if len(inputs[id]) is not 2 else len(inputs[id][0])
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents), use_cuda=True)
        for ix, modality in enumerate(inputs):
            if modality is not None:
                mod_mu, mod_logvar = self.vaes[ix].enc(modality.to("cuda") if not isinstance(modality, list) else modality)
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

class MoPOE(POE):
    def __init__(self, encoders, decoders, data_paths,  feature_dims, mod_types, n_latents, test_split, batch_size):
        """
        Multimodal Variaional Autoencoder with Generalized Multimodal Elbo https://github.com/thomassutter/MoPoE

        :param encoders: list of encoder names (strings) as listed in config
        :type encoders: list
        :param decoders: list of decoder names (strings) as listed in config
        :type decoders: list
        :param data_paths: list of data paths for all modalities
        :type data_paths: list
        :param feature_dims: list of modality-specific feature dimensions as listed in config
        :type feature_dims: list
        :param mod_types: list of modality types (strings) from config
        :type mod_types: list
        :param n_latents: list of latent dimensionalities from config
        :type n_latents: list
        :param test_split: fraction of the data to be used for validation
        :type test_split: float
        :param batch_size: batch size
        :type batch_size: int
        """
        self.modelName = 'mopoe'
        super(MoPOE, self).__init__(encoders, decoders, data_paths,  feature_dims, mod_types, n_latents, test_split, batch_size)
        self.n_latents = n_latents
        self.modelName = "mopoe"
        self.subsets = [[x] for x in self.vaes] + list(combinatorial([x for x in self.vaes]))

    def infer(self, inputs, num_samples=None):
        mu, logvar = [None] * len(self.vaes), [None] * len(self.vaes)
        for ix, modality in enumerate(inputs):
            if modality is not None:
                mod_mu, mod_logvar = self.vaes[ix].enc(modality.to("cuda") if not isinstance(modality, list) else modality)
                mu[ix] = mod_mu.unsqueeze(0)
                logvar[ix] = mod_logvar.unsqueeze(0)
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
            mus = torch.cat((mus.squeeze(1), torch.zeros(1, mus.shape[-2], self.n_latents).cuda()), dim=0)
            logvars = torch.cat((logvars.squeeze(1), torch.zeros(1, mus.shape[-2], self.n_latents).cuda()), dim=0)
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


class DMVAE(MMVAE):
    def __init__(self, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size):
        """
        Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations https://github.com/seqam-lab/DMVAE

        :param encoders: list of encoder names (strings) as listed in config
        :type encoders: list
        :param decoders: list of decoder names (strings) as listed in config
        :type decoders: list
        :param data_paths: list of data paths for all modalities
        :type data_paths: list
        :param feature_dims: list of modality-specific feature dimensions as listed in config
        :type feature_dims: list
        :param mod_types: list of modality types (strings) from config
        :type mod_types: list
        :param n_latents: list of latent dimensionalities from config
        :type n_latents: list
        :param test_split: fraction of the data to be used for validation
        :type test_split: float
        :param batch_size: batch size
        :type batch_size: int
        """
        self.modelName = 'dmvae'
        super(DMVAE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size)
        self.n_latents = n_latents
        self.qz_x = dist.Normal


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
                    if "transformer" in vae.dec_name.lower():
                        px_zs[d][e] = vae.px_z(*vae.dec([zs, x[d][1]] if x[d] is not None else [zs, None]))
                    else:
                        px_zs[d][e] = vae.px_z(*vae.dec(zs))
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
