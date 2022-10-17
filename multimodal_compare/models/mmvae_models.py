# Multi-modal model specification
from models.mmvae_base import TorchMMVAE
import torch, copy
import torch.distributions as dist
import torch.nn as nn
from utils import combinatorial, unpack_vae_outputs, log_joint, log_batch_marginal, get_all_pairs, subsample_input_modalities, find_out_batch_size
from torch.autograd import Variable
from models.NetworkTypes import VaeOutput


class MOE(TorchMMVAE):
    def __init__(self, vaes, obj_config, model_config=None):
        """
        Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(**obj_config)
        self.model_config = model_config
        self.vaes = nn.ModuleDict(vaes)
        self.modelName = 'moe'
        self.prior_dist = dist.Normal
        self.pz = dist.Normal

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

    def objective(self, data):
        """
        Objective function for MoE

        :param data: input data with modalities as keys
        :type data: dict
        :return: loss calculated using self.obj_fn
        :rtype: torch.tensor
        """
        output_dict = self.forward(data)
        qz_xs, px_zs, zss = [], [[None for _ in range(len(data.keys()))] for _ in range(len(data.keys()))], []
        for ind, mod in enumerate(output_dict.keys()):
            qz_xs.append(output_dict[mod].encoder_dists)
            zss.append(output_dict[mod].latent_samples)
            px_zs[ind][ind] = output_dict[mod].decoder_dists[0]
            pos = 0 if ind == 1 else 1
            px_zs[ind][pos] = output_dict[mod].decoder_dists[1]
        lpx_zs, klds = [], []
        for r, qz_x in enumerate(qz_xs):
            kld = self.obj_fn.calc_kld(qz_x, self.pz(*self.vaes["mod_{}".format(r + 1)]._pz_params.cuda()))
            klds.append(kld.sum(-1))
            for d in range(len(px_zs)):
                self.obj_fn.set_ltype(self.vaes["mod_{}".format(d + 1)].ltype)
                lpx_z = self.obj_fn.recon_loss_fn(px_zs[d][d], data["mod_{}".format(d + 1)]["data"]).view(*px_zs[d][d].batch_shape[:1],
                                                                                           -1)
                lpx_z = (lpx_z * self.vaes["mod_{}".format(d + 1)].llik_scaling).sum(-1)
                if d == r:
                    lwt = torch.tensor(0.0).cuda()
                else:
                    zs = zss[d]["latents"].detach()
                    qz_x.log_prob(zs)[torch.isnan(qz_x.log_prob(zs))] = 0
                    lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)[0][0]
                lpx_zs.append((lwt.exp() * lpx_z))
        d = {"lpx_z":torch.stack([lp for lp in lpx_zs if lp.sum() != 0]), "kld": torch.stack(klds), "qz_x":qz_xs, "zs": zss, "pz":self.pz, "pz_params":self.pz_params}
        obj = self.obj_fn.calculate_loss(d)
        if self.obj_fn.obj_name == "elbo":
            obj["loss"] = (1 / len(self.vaes)) * obj["loss"]
        return obj

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
        for modality, qz in qz_xs.items():
            if qz is not None:
                qz_xs[modality] = self.vaes[modality].qz_x(*list(qz))
                z = self.vaes[modality].qz_x(*qz).rsample(torch.Size([K]))
                zs[modality] = {"latents": z, "masks": None}
            else:
                zs[modality] = {"latents": None, "masks": None}
        # decode the samples
        px_zs = self.decode(zs)
        for modality, px_z in px_zs.items():
            if px_z[0]:
                px_zs[modality] = [dist.Normal(*px_z[0])]
        missing, filled = self.get_missing_modalities(qz_xs)
        assert len(filled) > 0, "at least one modality must be present for forward call"
        for mod_name in missing:
            zs[mod_name] = zs[filled[0]]
            px_zs[mod_name] = [dist.Normal(*self.vaes[mod_name].dec(zs[filled[0]]))]
        for modality, z in zs.items():
            for mod_vae, vae in self.vaes.items():
                if mod_vae != modality:  # fill-in off-diagonal
                    px_zs[mod_vae].append(vae.px_z(*vae.dec(z)))
        return self.make_output_dict(qz_xs, px_zs, zs)

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
    def __init__(self, vaes, obj_config, model_config=None):
        """
        Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(**obj_config)
        self.vaes = nn.ModuleDict(vaes)
        self.model_config = model_config
        self.modelName = 'poe'
        self.pz = dist.Normal
        self.prior_dist = dist.Normal

    def objective(self, mods):
        """
        Objective function for PoE

        :param data: input data with modalities as keys
        :type data: dict
        :return: loss calculated using self.obj_fn
        :rtype: torch.tensor
        """
        lpx_zs, klds, losses = [[] for _ in range(len(mods.keys()))], [], []
        mods_inputs = subsample_input_modalities(mods)
        for m, mods_input in enumerate(mods_inputs):
            output_dic = self.forward(mods_input)
            qz_xs, zss, px_zs, _ = unpack_vae_outputs(output_dic)
            kld = self.obj_fn.calc_kld(qz_xs[0], self.pz(*self.pz_params.to("cuda")))
            klds.append(kld.sum(-1))
            loc_lpx_z = []
            for mod in output_dic.keys():
                px_z = output_dic[mod].decoder_dists[0]
                self.obj_fn.set_ltype(self.vaes[mod].ltype)
                lpx_z = (self.obj_fn.recon_loss_fn(px_z, mods[mod]["data"]) * self.vaes[mod].llik_scaling).sum(-1)
                loc_lpx_z.append(lpx_z)
                if mod == "mod_{}".format(m + 1):
                    lpx_zs[m].append(lpx_z)
            d = {"lpx_z": torch.stack(loc_lpx_z).sum(0), "kld": kld.sum(-1), "qz_x": qz_xs, "zs": zss, "pz": self.pz, "pz_params": self.pz_params}
            losses.append(self.obj_fn.calculate_loss(d)["loss"])
        ind_losses = [-torch.stack(m).sum() / self.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs)]
        obj = {"loss": torch.stack(losses).sum(), "reconstruction_loss": ind_losses, "kld": torch.stack(klds).mean(0).sum()}
        return obj


    def forward(self, inputs, K=1):
        """
        Forward pass that takes input data and outputs a dict with  posteriors, reconstructions and latent samples
        :param inputs: input data, a dict of modalities where missing modalities are replaced with None
        :type inputs: dict
        :param K: sample K samples from the posterior
        :type K: int
        :return: dict where keys are modalities and values are a named tuple
        :rtype: dict
        """
        mu, logvar, single_params = self.modality_mixing(inputs, K)
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        qz_d, px_d, z_d = {}, {}, {}
        for mod, vae in self.vaes.items():
            px_d[mod] = vae.px_z(*vae.dec({"latents": z, "masks": None}))
        for key in inputs.keys():
            qz_d[key] = qz_x
            z_d[key] = {"latents": z, "masks": None}
        return self.make_output_dict(qz_d, px_d, z_d)

    def modality_mixing(self, x, K=1):
        """
        Inference module, calculates the joint posterior
        :param inputs: input data, a dict of modalities where missing modalities are replaced with None
        :type inputs: dict
        :param K: sample K samples from the posterior
        :type K: int
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        batch_size = find_out_batch_size(x)
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.vaes["mod_1"].n_latents), use_cuda=True)
        for m, vae in self.vaes.items():
            if x[m]["data"] is not None:
                mod_mu, mod_logvar = vae.enc(x[m])
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)
        mu_before, logvar_before = mu, logvar
        # product of experts to combine gaussians
        mu, logvar = super(POE, POE).product_of_experts(mu, logvar)
        return mu, logvar, [mu_before[1:], logvar_before[1:]]


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


class MoPOE(TorchMMVAE):
    def __init__(self, vaes, obj_config, model_config=None):
        """
        Multimodal Variational Autoencoder with Generalized Multimodal Elbo https://github.com/thomassutter/MoPoE

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(**obj_config)
        self.vaes = nn.ModuleDict(vaes)
        self.model_config = model_config
        self.modelName = 'mopoe'
        self.pz = dist.Normal
        self.subsets = [[x] for x in self.vaes] + list(combinatorial([x for x in self.vaes]))
        self.prior_dist = dist.Normal

    def objective(self, mods):
        """
        Objective function for MoPoE. Computes GENERALIZED MULTIMODAL ELBO https://arxiv.org/pdf/2105.02470.pdf

        :param data: input data with modalities as keys
        :type data: dict
        :return: loss calculated using self.obj_fn
        :rtype: torch.tensor
        """
        qz_xs, zss, px_zs, single_latents = unpack_vae_outputs(self.forward(mods))
        lpx_zs, klds = [], []
        uni_mus, uni_logvars = single_latents
        uni_dists = [dist.Normal(*[mu, logvar]) for mu, logvar in zip(uni_mus, uni_logvars)]
        for r, px_z in enumerate(px_zs):
            tag = "mod_{}".format(r + 1)
            self.obj_fn.set_ltype(self.vaes["mod_{}".format(r + 1)].ltype)
            lpx_z = self.obj_fn.recon_loss_fn(px_z[0], mods[tag]["data"]).cuda() * self.vaes["mod_{}".format(r+1)].llik_scaling
            lpx_zs.append(lpx_z.sum(-1))
        rec_loss = torch.stack(lpx_zs).sum() / len(lpx_zs)
        rec_loss = Variable(rec_loss, requires_grad=True)
        group_divergence = self.obj_fn.calc_klds(qz_xs, self)
        kld_mods = self.obj_fn.calc_klds(uni_dists, self)
        kld_weighted = (torch.stack(kld_mods).sum(0) + torch.stack(group_divergence).sum(0)).sum()
        d = {"lpx_z": rec_loss, "kld": kld_weighted, "qz_xs": qz_xs, "zs": zss, "pz": self.pz, "pz_params": self.pz_params}
        obj = self.obj_fn.calculate_loss(d)
        individual_losses = [-m.sum() / self.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs)]
        obj["reconstruction_loss"] = individual_losses
        return obj

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
        mu, logvar, single_latents = self.modality_mixing(inputs, K)
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        qz_d, px_d, z_d = {}, {}, {}
        z_d["joint"] = {"latents": z, "masks": None}
        for mod, vae in self.vaes.items():
            px_d[mod] = vae.px_z(*vae.dec({"latents": z, "masks": None}))
        for key in inputs.keys():
            qz_d[key] = qz_x
            z_d[key] = {"latents": z, "masks": None}
        return self.make_output_dict(qz_d, px_d, z_d, single_latents)

    def modality_mixing(self, x, num_samples=None):
        mu, logvar = [None] * len(self.vaes), [None] * len(self.vaes)
        for m, vae in enumerate(self.vaes.values()):
            tag = "mod_{}".format(m + 1)
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
        single_latents = {}
        for i, mu in enumerate(mus[:2]):
            single_latents["mod_{}".format(i+1)] = [mu, torch.clamp(logvars, min=0.0001)[i]]
        return joint_mu, joint_logvar, single_latents

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
            mus = torch.cat((mus.squeeze(1), torch.zeros(1, mus.shape[-2], self.vaes['mod_1'].n_latents).cuda()), dim=0)
            logvars = torch.cat((logvars.squeeze(1), torch.zeros(1, mus.shape[-2], self.vaes["mod_1"].n_latents).cuda()),
                                dim=0)
        mu_poe, logvar_poe = super(MoPOE, MoPOE).product_of_experts(mus, logvars)
        if len(mu_poe.shape) < 3:
            mu_poe = mu_poe.unsqueeze(0)
            logvar_poe = logvar_poe.unsqueeze(0)
        return [mu_poe, logvar_poe]

    def mixture_component_selection(self, mus, logvars, w_modalities=None):
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
    def __init__(self, vaes, obj_config, model_config=None):
        """
        Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations https://github.com/seqam-lab/DMVAE

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(**obj_config)
        self.model_config = model_config
        self.vaes = nn.ModuleDict(vaes)
        self.modelName = 'dmvae'
        self.pz = dist.Normal
        self.qz_x = dist.Normal


    def objective(self, mods):
        """
        Objective for the DMVAE model. Source: https://github.com/seqam-lab/

        :param data: input data with modalities as keys
        :type data: dict
        :return: loss calculated using self.obj_fn
        :rtype: torch.tensor
        """
        qz_xs, zss, px_zs, _ = unpack_vae_outputs(self.forward(mods))
        recons, kls, ind_losses = [], [], []
        for i in range(len(px_zs)):
            tag = "mod_{}".format(i + 1)
            for j in range(len(px_zs[i])):
                if j < len(px_zs[i]) - 1:
                    self.obj_fn.set_ltype(self.vaes[tag].ltype)
                    for p in px_zs[i][j]:
                        recons.append(self.obj_fn.recon_loss_fn(p, mods[tag]["data"]).cuda() * self.vaes[tag].llik_scaling)
                else:
                    recons.append(torch.tensor(0))
            for n in range(len(px_zs[i]) - 1):
                idxs = [2 + n, 4]
                log_pz = log_joint([px_zs[i][n], px_zs[n + 1]], [zss[i], zss[idxs[n]]])
                log_q_zCx = log_joint([qz_xs[i], qz_xs[idxs[n]]], [zss[i], zss[idxs[n]]])
                log_qz, _, log_prod_qzi = log_batch_marginal([qz_xs[i], qz_xs[idxs[n]]])
                kl = ((log_q_zCx - log_qz) * (log_qz - log_prod_qzi) * (log_prod_qzi - log_pz)).mean()
                kls.append(kl)
        # cross sampling
        for i in get_all_pairs(px_zs):
            tag = "mod_{}".format(i + 1)
            self.obj_fn.set_ltype(self.vaes[i].ltype)
            recons.append(self.obj_fn.loss_fn(px_zs[i[0]][0], mods[tag]["data"]).cuda() * self.vaes[tag].llik_scaling.mean())
            log_pz = log_joint([px_zs[i[0]][0], px_zs[i[0]][1]])
            log_q_zCx = log_joint([qz_xs[i[0]][0], qz_xs[i[0]][1]])
            log_qz, _, log_prod_qzi = log_batch_marginal([qz_xs[i[0]][0], qz_xs[i[0]][1]])
            kl = ((log_q_zCx - log_qz) * (log_qz - log_prod_qzi) * (log_prod_qzi - log_pz)).mean()
            kls.append(kl)
        for rec, kl in zip(recons, kls):
            d = {"lpx_z": rec, "kld": kl, "qz_xs": qz_xs, "zs": zss, "pz": self.pz, "pz_params": self.pz_params}
            ind_losses.append(self.obj_fn.calculate_loss(d)["loss"])
        obj = {"loss":torch.tensor(ind_losses).sum(), "reconstruction_loss": ind_losses, "kld": torch.stack(kls).mean(0).sum()}
        return obj

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
        qz_xs_shared, px_zs = [], [[None for _ in range(len(self.vaes) + 1)] for _ in range(len(self.vaes))]
        qz_xs_private = [None for _ in range(len(self.vaes))]
        # initialise cross-modal matrix
        for m, vae in enumerate(self.vaes.values()):
            if x["mod_{}".format(m+1)] is not None:
                mod_mu, mod_std = vae.enc(x["mod_{}".format(m+1)])
                qz_xs_private[m] = vae.qz_x(*[mod_mu[0], mod_std[0]])
                qz_xs_shared.append(vae.qz_x(*[mod_mu[1], mod_std[1]]))
        mu_joint, std_joint = self.product_of_experts(torch.stack([x.loc for x in qz_xs_shared]),
                                                      torch.stack([x.scale for x in qz_xs_shared]))
        joint_d = self.qz_x(*[mu_joint, std_joint])
        all_shared = qz_xs_shared + [joint_d]
        zss = []
        for d, vae in enumerate(self.vaes.values()):
            for e, dist in enumerate(all_shared):
                zs_shared = dist.rsample(torch.Size([K]))
                zs_private = qz_xs_private[d].rsample(torch.Size([K]))
                zs = torch.cat([zs_private, zs_shared], -1)[0]
                zss.append(zs)
                px_zs[d][e] = vae.px_z(*vae.dec({"latents": zs, "masks": None}))
        output_dict = {}
        for modality in self.vaes.keys():
            output_dict[modality] = VaeOutput(encoder_dists=[qz_xs_private,all_shared], decoder_dists=px_zs,
                                              latent_samples=zss, single_latents=None)
        return output_dict

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