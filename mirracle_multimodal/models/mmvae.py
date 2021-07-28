# Base MMVAE class definition

from itertools import combinations

import torch
import torch.nn as nn

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df
from torch.autograd import Variable
import torch.distributions as dist


class MMVAE(nn.Module):
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        self.vaes = nn.ModuleList([vae(params, index) for index, vae in enumerate(vaes)])
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pz_params = None  # defined in subclass

    @property
    def pz_params(self):
        return self._pz_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons

    def analyse(self, data, K):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df

class MMVAE_P(nn.Module):
    """Multimodal Variational Autoencoder."""
    def __init__(self, prior_dist, params, *vaes):
            super(MMVAE_P, self).__init__()
            self.pz = prior_dist
            self.vaes = nn.ModuleList([vae(params, index) for index, vae in enumerate(vaes)])
            self.modelName = None  # filled-in per sub-class
            self.params = params
            self._pz_params = None  # defined in subclass
            self.experts = ProductOfExperts()

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
        mu, logvar = prior_expert((1, next(x for x in inputs if x is not None).shape[0], self.params.latent_dim), use_cuda=True)
        for ix, modality in enumerate(inputs):
            if modality is not None:
                mod_mu, mod_logvar = self.vaes[ix].enc(modality.to("cuda"))
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)

        mu_before, logvar_before = mu, logvar
        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar, [mu_before[1:], logvar_before[1:]]

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions
            recons = [get_mean(r) for r in px_zs]
        return recons

    def analyse(self, data, K):
        self.eval()
        with torch.no_grad():
            zss = []
            qz_xs, _, _= self.forward(data, both_qz=True)
            for i in qz_xs:
                zss.append(i.rsample(torch.Size([K])))
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = pd_var
        return pd_mu, pd_logvar

def prior_expert(size, use_cuda=False):
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