# Multi-modal model specification
from models.mmvae_base import TorchMMVAE
import torch
import torch.distributions as dist
import torch.nn as nn
from utils import combinatorial, log_joint, log_batch_marginal, get_all_pairs, subsample_input_modalities, find_out_batch_size
from torch.autograd import Variable
from itertools import chain, combinations

class MOE(TorchMMVAE):
    def __init__(self, vaes:list, n_latents:int, obj_config:dict, model_config=None):
        """
        Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae
        
        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param n_latents: dimensionality of the (shared) latent space
        :type n_latents: int
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(vaes, n_latents, **obj_config)
        self.model_config = model_config
        self.modelName = 'moe'

    @property
    def pz_params(self):
        return nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, self.n_latents), requires_grad=True)  # logvar
        ])

    def objective(self, data):
        """
        Objective function for MoE

        :param data: input data with modalities as keys
        :type data: dict
        :return obj: dictionary with the obligatory "loss" key on which the model is optimized, plus any other keys that you wish to log
        :rtype obj: dict
        """
        output = self.forward(data, K=self.K)
        out_d = output.unpack_values()
        lpx_zs, klds = [], []
        for r, qz_x in enumerate(out_d["encoder_dist"]):
            kld = self.obj_fn.calc_kld(qz_x, self.pz(*self.vaes["mod_{}".format(r + 1)]._pz_params.cuda()))
            klds.append(kld.sum(-1))
            self.obj_fn.set_ltype(self.vaes["mod_{}".format(r + 1)].ltype)
            lpx_z = (self.obj_fn.recon_loss_fn(out_d["decoder_dist"][r], data["mod_{}".format(r + 1)], K=self.K).view(*out_d["decoder_dist"][r].batch_shape[:1], -1)
                     * self.vaes["mod_{}".format(r + 1)].llik_scaling).sum(-1)
            lpx1 = (torch.tensor(0.0).cuda().exp() * lpx_z)
            for key, cros_l in output.mods["mod_{}".format(r+1)].cross_decoder_dist.items():
                lpx_z = (self.obj_fn.recon_loss_fn(cros_l, data["mod_{}".format(r + 1)], K=self.K).view(
                    *cros_l.batch_shape[:1], -1)
                         * self.vaes["mod_{}".format(r + 1)].llik_scaling).sum(-1)
                zs = out_d["latent_samples"][int(key.split("_")[-1])-1]["latents"].detach()
                q = out_d["encoder_dist"][int(key.split("_")[-1])-1]
                qz_x.log_prob(zs)[torch.isnan(qz_x.log_prob(zs))] = 0
                lwt = (qz_x.log_prob(zs) - q.log_prob(zs).detach()).sum(-1).reshape(-1)
                lwt = lwt #/abs(torch.max(lwt))
                if self.obj_fn.obj_name == "elbo":
                    lpx_zs.append(lpx1)
                    lpx_zs.append((lwt.exp() * lpx_z))
                else:
                    lpx_zs.append([lpx1, lpx_z])
        lpx = torch.stack([lp for lp in lpx_zs if lp.sum() != 0]) if not isinstance(lpx_zs[0], list) else lpx_zs
        d = {"lpx_z":lpx, "kld": torch.stack(klds), "qz_x":out_d["encoder_dist"], "zs": out_d["latent_samples"], "pz":self.pz, "pz_params":self.pz_params, "K":self.K}
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
        missing, filled = self.get_missing_modalities(x)
        assert len(filled) > 0, "at least one modality must be present for forward call"
        qz_xs = self.encode(x)
        zs, cross_px_zs, qzs = {}, {}, {}
        for modality, qz in qz_xs.items():
            qzs[modality] = self.vaes[modality].qz_x(*qz["shared"]) if qz["shared"] is not None else None
            if qz["shared"] is not None:
                qz_xs[modality] = self.vaes[modality].qz_x(*list(qz["shared"]))
                z = self.vaes[modality].qz_x(*qz["shared"]).rsample(torch.Size([K]))
                zs[modality] = {"latents": z, "masks": x[modality]["masks"]}
            else:
                zs[modality] = {"latents": None, "masks": x[modality]["masks"]}
        # decode the samples
        px_zs = self.decode(zs)
        for modality, px_z in px_zs.items():
            if px_z is not None:
                px_zs[modality] = dist.Normal(*px_z)
        for mod_name in missing:
            zs[mod_name] = zs[filled[0]]
            zs[mod_name]["masks"] = x[mod_name]["masks"]
            px_zs[mod_name] = dist.Normal(*self.vaes[mod_name].dec(zs[filled[0]]))
        for modality, z in zs.items():
            for mod_vae, vae in self.vaes.items():
                if mod_vae != modality:  # fill-in off-diagonal
                    z["masks"] = x[mod_vae]["masks"]
                    cross_px_zs[mod_vae] = {modality:vae.px_z(*vae.dec(z))}
        return self.make_output_dict(qzs, px_zs, zs, cross_decoder_dist=cross_px_zs)

    def reconstruct(self, data, runPath, epoch):
        """
        Reconstruct data for individual experts

        :param data: list of input modalities
        :type data: list
        :param runPath: path to save data to
        :type runPath: str
        :param epoch: current epoch to name the data
        :type epoch: str
        """
        recons_mat = super(MOE, self).reconstruct([d for d in data])
        self.process_reconstructions(recons_mat, data, epoch, runPath)


class POE(TorchMMVAE):
    def __init__(self, vaes:list, n_latents:int, obj_config:dict, model_config=None):
        """
        Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param n_latents: dimensionality of the (shared) latent space
        :type n_latents: int
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(vaes, n_latents, **obj_config)
        self.model_config = model_config
        for vae in self.vaes:
            assert vae.prior_str == "normal", "PoE mixing only works with normal (gaussian) priors! adjust the config"
        self.modelName = 'poe'

    def objective(self, mods):
        """
        Objective function for PoE

        :param data: input data with modalities as keys
        :type data: dict
        :return obj: dictionary with the obligatory "loss" key on which the model is optimized, plus any other keys that you wish to log
        :rtype obj: dict
        """
        lpx_zs, klds, losses = [[] for _ in range(len(mods.keys()))], [], []
        mods_inputs = subsample_input_modalities(mods)
        for m, mods_input in enumerate(mods_inputs):
            output = self.forward(mods_input)
            output_dic = output.unpack_values()
            kld = self.obj_fn.calc_kld(output_dic["joint_dist"][0], self.pz(*self.pz_params.to("cuda")))
            klds.append(kld.sum(-1))
            loc_lpx_z = []
            for mod in output.mods.keys():
                px_z = output.mods[mod].decoder_dist
                self.obj_fn.set_ltype(self.vaes[mod].ltype)
                lpx_z = (self.obj_fn.recon_loss_fn(px_z, mods[mod]) * self.vaes[mod].llik_scaling).sum(-1)
                loc_lpx_z.append(lpx_z)
                if mod == "mod_{}".format(m + 1):
                    lpx_zs[m].append(lpx_z)
            d = {"lpx_z": torch.stack(loc_lpx_z).sum(0), "kld": kld.sum(-1), "qz_x": output_dic["encoder_dist"], "zs": output_dic["latent_samples"], "pz": self.pz, "pz_params": self.pz_params}
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
        mu, logvar, single_params = self.modality_mixing(inputs)
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([K]))
        qz_d, px_d, z_d = {}, {}, {}
        for mod, vae in self.vaes.items():
            px_d[mod] = vae.px_z(*vae.dec({"latents": z, "masks": inputs[mod]["masks"]}))
        for key in inputs.keys():
            qz_d[key] = qz_x
            z_d[key] = {"latents": z, "masks": inputs[key]["masks"]}
        return self.make_output_dict(single_params, px_d, z_d, joint_dist=qz_d)

    def modality_mixing(self, x):
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
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents), use_cuda=True)
        single_params = {}
        for m, vae in self.vaes.items():
            if x[m]["data"] is not None:
                mod_mu, mod_logvar = vae.enc(x[m])
                single_params[m] = dist.Normal(*[mod_mu, mod_logvar])
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)
        # product of experts to combine gaussians
        mu, logvar = super(POE, POE).product_of_experts(mu, logvar)
        return mu, logvar, single_params


    def prior_expert(self, size, use_cuda=False):
        """
        Universal prior expert. Here we use a spherical Gaussian: N(0, 1).

        :param size: dimensionality of the Gaussian
        :type size: int
        :param use_cuda: cast CUDA on variables
        :type use_cuda: boolean
        :return: mean and logvar of the expert
        :rtype: tuple
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if use_cuda:
            mu, logvar = mu.to("cuda"), logvar.to("cuda")
        return mu, logvar


class MoPOE(TorchMMVAE):
    def __init__(self, vaes:list, n_latents:int, obj_config:dict, model_config=None):
        """
        Multimodal Variational Autoencoder with Generalized Multimodal Elbo https://github.com/thomassutter/MoPoE

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param n_latents: dimensionality of the (shared) latent space
        :type n_latents: int
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(vaes, n_latents, **obj_config)
        self.model_config = model_config
        self.modelName = 'mopoe'
        self.subsets = [[x] for x in self.vaes] + list(combinatorial([x for x in self.vaes]))
        self.subsets = self.set_subsets()
        self.weights = None

    def set_subsets(self):
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        xs = list(self.vaes.keys())
        subsets_list = chain.from_iterable(combinations(xs, n) for n in range(len(xs)+1))
        subsets = dict()
        for k, mod_names in enumerate(subsets_list):
            mods = []
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.vaes[mod_name])
            key = '_'.join(sorted(mod_names))
            subsets[key] = mods
        subsets.pop("", None)
        return subsets

    def objective(self, mods):
        """
        Objective function for MoPoE. Computes GENERALIZED MULTIMODAL ELBO https://arxiv.org/pdf/2105.02470.pdf

        :param data: input data with modalities as keys
        :type data: dict
        :return obj: dictionary with the obligatory "loss" key on which the model is optimized, plus any other keys that you wish to log
        :rtype obj: dict
        """
        output = self.forward(mods)
        out_unpacked = output.unpack_values()
        lpx_zs, klds = [], []
        dists = out_unpacked["encoder_dist"] + [out_unpacked["joint_dist"][0]]
        group_divergence = self.obj_fn.weighted_group_kld(dists, self,  (1/len(dists))*torch.ones(len(dists)).to("cuda"))
        for r, px_z in enumerate(out_unpacked["decoder_dist"]):
             tag = "mod_{}".format(r + 1)
             self.obj_fn.set_ltype(self.vaes["mod_{}".format(r + 1)].ltype)
             lpx_z = self.obj_fn.recon_loss_fn(px_z, mods[tag]).cuda() * self.vaes["mod_{}".format(r+1)].llik_scaling
             lpx_zs.append(lpx_z.sum(-1))
        d = {"lpx_z": torch.stack(lpx_zs).sum(0).mean(), "kld": group_divergence[0], "qz_x": out_unpacked["encoder_dist"], "zs": out_unpacked["latent_samples"], "pz": self.pz,
             "pz_params": self.pz_params}
        obj = self.obj_fn.calculate_loss(d)
        ind_losses = [-m / self.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs)]
        obj["reconstruction_loss"] = ind_losses
        return obj

    def modality_mixing(self, input_batch):
        latents = dict()
        enc_mods = self.encode(input_batch)
        latents['modalities'] = enc_mods
        mus = torch.Tensor().to("cuda")
        logvars = torch.Tensor().to("cuda")
        distr_subsets = dict()
        for k, s_key in enumerate(self.subsets.keys()):
            mods = self.subsets[s_key]
            mus_subset = torch.Tensor().to("cuda")
            logvars_subset = torch.Tensor().to("cuda")
            mods_avail = True
            for m, mod in enumerate(mods):
                if mod.modelName in input_batch.keys() and input_batch[mod.modelName]["data"] is not None:
                    mus_subset = torch.cat((mus_subset, enc_mods[mod.modelName]["shared"][0].unsqueeze(0)), dim=0)
                    logvars_subset = torch.cat((logvars_subset,  enc_mods[mod.modelName]["shared"][1].unsqueeze(0)),  dim=0)
                else:
                    mods_avail = False
            if mods_avail:
                s_mu, s_logvar = self.poe_fusion(mus_subset, logvars_subset)
                distr_subsets[s_key] = [s_mu, s_logvar]
                mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)
        self.weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to("cuda")
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, self.weights)
        latents['joint'] = [joint_mu.squeeze(0), joint_logvar.squeeze(0)]
        latents['subsets'] = distr_subsets
        return latents

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
        latents = self.modality_mixing(inputs)
        qz_d, px_d, z_d, qz_joint = {}, {}, {}, {}
        for mod, vae in self.vaes.items():
            qz_d[mod] = dist.Normal(*latents["modalities"][mod]["shared"]) if latents["modalities"][mod]["shared"] is not None else None
            qz_joint[mod] = dist.Normal(*latents["joint"])
            z = qz_joint[mod].rsample(torch.Size([K]))#qz_d[mod].rsample(torch.Size([1])) if latents["modalities"][mod]["shared"] is not None \
                #else qz_joint[mod].rsample(torch.Size([1]))
            z_d[mod] = {"latents": z, "masks": inputs[mod]["masks"]}
            px_d[mod] = vae.px_z(*vae.dec(z_d[mod]))
        return self.make_output_dict(qz_d, px_d, z_d, qz_joint)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def reweight_weights(self, w):
        return w / w.sum()

    def moe_fusion(self, mus, logvars, weights):
        weights = self.reweight_weights(weights)
        mu_moe, logvar_moe = self.mixture_component_selection(mus, logvars, weights)
        return [mu_moe, logvar_moe]

    def poe_fusion(self, mus, logvars):
        if mus.shape[0] == len(self.vaes):
            mus = torch.cat((mus.squeeze(1), torch.zeros(1, mus.shape[-2], self.n_latents).cuda()), dim=0)
            logvars = torch.cat((logvars.squeeze(1), torch.zeros(1, mus.shape[-2], self.n_latents).cuda()),
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
    def __init__(self, vaes:list, n_latents:int, obj_config:dict, model_config=None):
        """
        Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations https://github.com/seqam-lab/DMVAE

        :param vaes: list of modality-specific vae objects
        :type vaes: list
        :param n_latents: dimensionality of the (shared) latent space
        :type n_latents: int
        :param obj_cofig: config with objective-specific parameters (obj name, beta.)
        :type obj_config: dict
        :param model_cofig: config with model-specific parameters
        :type model_config: dict
        """
        super().__init__(vaes, n_latents, **obj_config)
        self.model_config = model_config
        self.modelName = 'dmvae'
        assert self.latent_factorization, "DMVAE requires private_latents in the config"

    def objective(self, mods):
        """
        Objective for the DMVAE model. Source: https://github.com/seqam-lab/

        :param data: input data with modalities as keys
        :type data: dict
        :return obj: dictionary with the obligatory "loss" key on which the model is optimized, plus any other keys that you wish to log
        :rtype obj: dict
        """
        output_whole = self.forward(mods)
        losses, ind_losses, klds = [], [], []
        for mod, output in output_whole.mods.items():
            self.obj_fn.set_ltype(self.vaes[mod].ltype)
            lpx_z = (self.obj_fn.recon_loss_fn(output.decoder_dist, mods[mod]) * self.vaes[mod].llik_scaling).sum(-1)
            kld = self.obj_fn.calc_kld(output.encoder_dist, self.pz(*self.pz_params.to("cuda")))
            kld_poe = self.obj_fn.calc_kld(output.joint_dist, self.pz(*self.pz_params.to("cuda")))
            lpx_z_poe = (self.obj_fn.recon_loss_fn(output.joint_decoder_dist, mods[mod]) * self.vaes[mod].llik_scaling).sum(-1)
            lpx_zs_cross = []
            klds_priv = []
            for k, r in output.cross_decoder_dist.items():
                lpx_zs_cross.append((self.obj_fn.recon_loss_fn(r, mods[mod]) * self.vaes[mod].llik_scaling).sum(-1))
                klds_priv.append(self.obj_fn.calc_kld(output.enc_dist_private, self.pz(*self.vaes[mod].pz_params_private)))
            loss = self.obj_fn.calculate_loss({"lpx_z": lpx_z, "kld": kld.sum(-1)})["loss"] + self.obj_fn.calculate_loss({"lpx_z": lpx_z_poe, "kld": kld_poe})["loss"] \
                   + self.obj_fn.calculate_loss({"lpx_z": torch.stack(lpx_zs_cross).sum(), "kld": torch.stack(klds_priv).sum(-1)})["loss"]
            losses.append(loss)
            ind_losses.append(lpx_z)
            klds.append(kld)
        ind_losses_reweighted = [-(m).sum() / self.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(ind_losses)]
        obj = {"loss":torch.stack(losses).sum(), "reconstruction_loss": ind_losses_reweighted, "kld": torch.stack(klds).mean(0).sum()}
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
        enc_d = self.encode(x)
        mu_joint, std_joint = self.product_of_experts(torch.stack([i["shared"][0] for i in enc_d.values() if i["shared"] is not None]),
                                                      torch.stack([i["shared"][1] for i in enc_d.values()if i["shared"] is not None]))
        joint_d = self.qz_x(*[mu_joint, std_joint])
        joint_dist, qz_xs, qz_private, zss, px_zs, joint_px_zs, cross_px_zs = {}, {}, {}, {}, {}, {}, {}
        for mod in self.vaes.keys():
            joint_dist[mod] = joint_d
            qz_xs[mod] = self.qz_x(*enc_d[mod]["shared"]) if enc_d[mod]["shared"] is not None else None
            qz_private[mod] = self.qz_x(*enc_d[mod]["private"]) if enc_d[mod]["private"] is not None else None
        z_joint = joint_d.rsample(torch.Size([K]))
        # decode from all dists
        for mod in self.vaes.keys():
           z_shared = qz_xs[mod].rsample(torch.Size([1])) if qz_xs[mod] is not None \
                   else qz_xs[self.get_missing_modalities(x)[1][0]].rsample(torch.Size([1])).cuda()
           z_private = qz_private[mod].rsample(torch.Size([1])) if qz_private[mod] is not None \
                else self.qz_x(*(torch.zeros(z_joint.shape[1], self.vaes[mod].private_latents),
                                 torch.ones(z_joint.shape[1], self.vaes[mod].private_latents))).rsample(torch.Size([1])).cuda()
           zss[mod] = {"latents": z_shared, "masks": x[mod]["masks"]}
           px_zs[mod] = self.vaes[mod].px_z(*self.vaes[mod].dec({"latents": torch.cat([z_shared, z_private], -1), "masks": x[mod]["masks"]}))
           joint_px_zs[mod] = self.vaes[mod].px_z(*self.vaes[mod].dec({"latents": torch.cat([z_joint, z_private], -1),
                                                                       "masks": x[mod]["masks"]}))
           cross_px_zs[mod] = {}
           for m in self.get_remaining_mods_data(qz_xs, mod):
               z_shared = qz_xs[m].rsample(torch.Size([1]))
               cross_px_zs[mod][m] = self.vaes[mod].px_z(*self.vaes[mod].dec({"latents": torch.cat([z_shared, z_private], -1),
                                                                       "masks": x[mod]["masks"]}))
        return self.make_output_dict(qz_xs, px_zs, zss, joint_dist, qz_private, None, joint_px_zs, cross_px_zs)

    def get_remaining_mods_data(self, qz_xs:dict, exclude_mod:str):
        all_keys = [k for k in qz_xs.keys() if qz_xs[k] is not None]
        if exclude_mod in all_keys:
            all_keys.remove(exclude_mod)
        return all_keys

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
