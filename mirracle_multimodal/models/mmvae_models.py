# Multi-modal model specification
import torch
import torch.distributions as dist
from .mmvae_base import MMVAE
from torch.autograd import Variable


class MoPOE(MMVAE):
    """Multimodal Variaional Autoencoder with Generalized Multimodal Elbo https://github.com/thomassutter/MoPoE"""
    def __init__(self, encoders, decoders, data_paths,  feature_dims, n_latents, batch_size):
        self.modelName = 'mopoe'
        super(MoPOE, self).__init__(dist.Normal, encoders, decoders, data_paths,  feature_dims, n_latents, batch_size)
        self.n_latents = n_latents

    def inference(self, input_batch, num_samples=None):
        if num_samples is None:
            num_samples = self.batch_size
        latents = dict()
        enc_mods = self.encode(input_batch)
        latents['modalities'] = enc_mods
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)
        distr_subsets = dict()
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != '':
                mods = self.subsets[s_key]
                mus_subset = torch.Tensor().to(self.flags.device)
                logvars_subset = torch.Tensor().to(self.flags.device)
                mods_avail = True
                for m, mod in enumerate(mods):
                    if mod.name in input_batch.keys():
                        mus_subset = torch.cat((mus_subset,
                                                enc_mods[mod.name][0].unsqueeze(0)), dim=0)
                        logvars_subset = torch.cat((logvars_subset,
                                                    enc_mods[mod.name][1].unsqueeze(0)), dim=0)
                    else:
                        mods_avail = False
                if mods_avail:
                    weights_subset = ((1 / float(len(mus_subset))) *
                                      torch.ones(len(mus_subset)).to(self.flags.device))
                    s_mu, s_logvar = self.modality_fusion(mus_subset,
                                                          logvars_subset,
                                                          weights_subset)
                    distr_subsets[s_key] = [s_mu, s_logvar]
                    if self.fusion_condition(mods, input_batch):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)
        if self.flags.modality_jsd:
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                                              self.flags.class_dim).to(self.flags.device)), dim=0)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device)),dim=0)
        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(self.flags.device)
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)
        latents['mus'] = mus
        latents['logvars'] = logvars
        latents['weights'] = weights
        latents['joint'] = [joint_mu, joint_logvar]
        latents['subsets'] = distr_subsets
        return latents

    def forward(self, input_batch):
        latents = self.inference(input_batch)
        results = dict()
        results['latents'] = latents
        results['group_distr'] = latents['joint']
        class_embeddings = self.reparameterize(latents['joint'][0],
                                                latents['joint'][1])
        div = self.calc_joint_divergence_moe(latents['mus'],
                                         latents['logvars'],
                                         latents['weights'])
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        results_rec = dict()
        enc_mods = latents['modalities']
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                m_s_mu, m_s_logvar = enc_mods[m_key + '_style']
                if self.flags.factorized_representation:
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar)
                else:
                    m_s_embeddings = None
                m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings))
                results_rec[m_key] = m_rec
        results['rec'] = results_rec
        return results

    def reweight_weights(self, w):
        w = w / w.sum()
        return w

    def divergence_static_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = weights.clone()
        weights = self.reweight_weights(weights)
        div_measures = self.divergence_static_prio(self.flags, mus, logvars, weights, normalization=self.batch_size)
        divs = dict()
        divs['joint_divergence'] = div_measures[0]
        divs['individual_divs'] = div_measures[1]
        divs['dyn_prior'] = None
        return divs

    def poe_fusion(self, mus, logvars):
        if (self.flags.modality_poe or mus.shape[0] == len(self.modalities.keys())):
            num_samples = mus[0].shape[0]
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                             self.flags.class_dim).to(self.flags.device)), dim=0)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                 self.flags.class_dim).to(self.flags.device)), dim=0)
        mu_poe, logvar_poe = self.poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def poe(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

    def calc_group_divergence_moe(self, flags, mus, logvars, weights, normalization=None):
        num_mods = mus.shape[0]
        num_samples = mus.shape[1]
        if normalization is not None:
            klds = torch.zeros(num_mods)
        else:
            klds = torch.zeros(num_mods, num_samples)
        klds = klds.to(flags.device)
        weights = weights.to(flags.device)
        for k in range(0, num_mods):
            kld_ind = self.calc_kl_divergence(mus[k, :, :], logvars[k, :, :],
                                         norm_value=normalization)
            if normalization is not None:
                klds[k] = kld_ind
            else:
                klds[k, :] = kld_ind
        if normalization is None:
            weights = weights.unsqueeze(1).repeat(1, num_samples)
        group_div = (weights * klds).sum(dim=0)
        return group_div, klds

    def calc_kl_divergence(self, mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
        if mu1 is None or logvar1 is None:
            KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
        else:
            KLD = -0.5 * (
                torch.sum(1 - logvar0.exp() / logvar1.exp() - (mu0 - mu1).pow(2) / logvar1.exp() + logvar0 - logvar1))
        if norm_value is not None:
            KLD = KLD / float(norm_value)
        return KLD


class MOE(MMVAE):
    """Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae"""
    def __init__(self, encoders, decoders, data_paths, feature_dims, n_latents, batch_size):
        self.modelName = 'moe'
        super(MOE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, n_latents, batch_size)


    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            if x[m] is not None:
                qz_x, px_z, zs = vae(x[m], K=K)
                qz_xs.append(qz_x)
                zss.append(zs)
                px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if "transformer" in self.vaes[d].dec_name.lower():
                        px_zs[e][d] = vae.px_z(*vae.dec([zs, x[d][1]] if x[d] is not None else [zs, None] ))
                    else:
                        px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def reconstruct(self, data, runPath, epoch, N=8):
        recons_mat = super(MOE, self).reconstruct([d for d in data])
        self.process_reconstructions(recons_mat, data, epoch, runPath)


class POE(MMVAE):
    """Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public"""
    def __init__(self, encoders, decoders, data_paths,  feature_dims, n_latents, batch_size):
        self.modelName = 'poe'
        super(POE, self).__init__(dist.Normal, encoders, decoders, data_paths,  feature_dims, n_latents, batch_size)
        self.n_latents = n_latents

    def forward(self, inputs, both_qz=False, K=1):
        mu, logvar, single_params = self.infer(inputs)
        recons = []
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        for ind, vae in enumerate(self.vaes):
            if "transformer" in vae.dec_name.lower():
               z_dec = [z, inputs[ind][1]] if inputs[ind] is not None else [z, None]
            else: z_dec = z
            recons.append(vae.px_z(*vae.dec(z_dec)))
        z = [z] if not both_qz else [z * len(recons)]
        qz_x = qz_x if not both_qz else [dist.Normal(*[single_params[0][0], single_params[1][0]]), dist.Normal(*[single_params[0][1], single_params[1][1]])]
        return qz_x, recons, z

    def infer(self,inputs):
        for x in inputs:
            if x is not None:
                batch_size = len(x) if len(x) is not 2 else len(x[0])
                break
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

    def product_of_experts(self, mu, logvar, eps=1e-8):
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

    def reconstruct(self, data, runPath, epoch, N=8):
        recons_mat = []
        for ix, i in enumerate(data):
            input_mat = [None] * len(data)
            input_mat[ix] = i[:N]
            rec = super(POE, self).reconstruct(input_mat)
            recons_mat.append(rec)
        self.process_reconstructions(recons_mat, data, epoch, runPath)

class DMBN(MMVAE):
    """Deep Modality Blending Networks https://github.com/myunusseker/Deep-Modality-Blending-Networks"""
    def __init__(self, encoders, decoders, data_paths, feature_dims, n_latents):
        self.modelName = 'dmbn'
        super(DMBN, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, n_latents)

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            if x[m] is not None:
                qz_x, px_z, zs = vae(x[m], K=K)
                qz_xs.append(qz_x)
                zss.append(zs)
                px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if self.vaes[d].dec_name == "Transformer":
                        px_zs[e][d] = vae.px_z(*vae.dec([zs, x[d][1]] if x[d] is not None else [zs, None] ))
                    else:
                        px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def reconstruct(self, data, runPath, epoch, N=8):
        recons_mat = super(MOE, self).reconstruct([d[:N] for d in data])
        self.process_reconstructions(recons_mat, data, epoch, runPath)
