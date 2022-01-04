# Multi-modal model specification - Mixture of Experts or Product of Experts
import torch
import torch.distributions as dist
from .mmvae_base import MMVAE
from torch.autograd import Variable

class MOE(MMVAE):
    def __init__(self, encoders, decoders, data_paths, feature_dims, n_latents):
        super(MOE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, n_latents)
        self.modelName = 'moe'

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
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def reconstruct(self, data, runPath, epoch, N=8):
        recons_mat = super(MOE, self).reconstruct([d[:N] for d in data])
        self.process_reconstructions(recons_mat, data, epoch, runPath)


class POE(MMVAE):
    def __init__(self, encoders, decoders, data_paths,  feature_dims, n_latents):
        super(POE, self).__init__(dist.Normal, encoders, decoders, data_paths,  feature_dims, n_latents)
        self.modelName = 'poe'

    def forward(self, inputs, both_qz=False, K=1):
        mu, logvar, single_params = self.infer(inputs)
        recons = []
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        for ind, vae in enumerate(self.vaes):
               recons.append(vae.px_z(*vae.dec(z)))
        z = [z] if not both_qz else [z * len(recons)]
        qz_x = qz_x if not both_qz else [dist.Normal(*[single_params[0][0], single_params[1][0]]), dist.Normal(*[single_params[0][1], single_params[1][1]])]
        return qz_x, recons, z

    def infer(self,inputs):
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, next(x for x in inputs if x is not None).shape[0], self.params["n_latents"]), use_cuda=True)
        for ix, modality in enumerate(inputs):
            if modality is not None:
                mod_mu, mod_logvar = self.vaes[ix].enc(modality.to("cuda"))
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
            input_mat = [None, None]
            input_mat[ix] = i[:N]
            rec = super(POE, self).reconstruct(input_mat)
            recons_mat.append(rec)
        self.process_reconstructions(recons_mat, data, epoch, runPath)
