# objectives of choice
import torch
from numpy import prod

from utils import log_mean_exp, is_multidata, kl_divergence


def loss_fn(output, target, ltype):
    if ltype == "bce":
        loss = -torch.nn.functional.binary_cross_entropy(output.loc.squeeze().cpu(), target.cpu(), reduction="sum").cuda()
    else:
        loss = output.log_prob(target).view(*output.batch_shape[:2], -1).sum(-1).sum(-1).sum(-1)
    return loss


def elbo(model, x, K=1, ltype="lprob"):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = loss_fn(px_z, x, ltype=ltype)
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return -(lpx_z.sum(-1) - kld.sum()).sum(), kld.sum(), -lpx_z.sum(), None


def m_moe_elbo(model, x, K=1, ltype="lprob"):
    """Importance-sampled m_elbo using the MoE approach with full supervision """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d][d], x[d], ltype=ltype).cuda() * model.vaes[d].llik_scaling
            if d == r:
                  lwt = torch.tensor(0.0).cuda()
            else:
                  zs = zss[d].detach()
                  lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1).sum()
            lpx_zs.append((lwt.exp() * lpx_z))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return -obj.sum(), torch.stack(klds).mean(0).sum(), -lpx_zs[0].sum() / model.vaes[0].llik_scaling, -lpx_zs[3].sum()

def m_poe_elbo(model, x, K, ltype="lprob"):
    """Importance-sampled m_elbo using the PoE approach with full supervision """

    qz_x, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    klds.append(kld.sum(-1))
    for d in range(len(px_zs)):
        lpx_z = loss_fn(px_zs[d], x[d], ltype=ltype)  * model.vaes[d].llik_scaling
        lwt = torch.tensor(0.0).cuda()
        lpx_zs.append(lwt.exp() * lpx_z)
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return -obj.sum(), torch.stack(klds).mean(0).sum(), -lpx_zs[0].sum() / model.vaes[0].llik_scaling, -lpx_zs[1].sum()

def m_poe_elbo_semi(model, x, K, ltype="lprob"):
    """Weakly supervised importance-sampled m_elbo using the PoE approach"""
    lpx_zs, klds, elbos = [[], []], [], []
    for m in range(len(x) + 1):
        mods = [None for _ in range(len(x))]
        if m == len(x):
            mods = x
        else:
            mods[m] = x[m]
        qz_x, px_zs, _ = model(mods)
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        loc_lpx_z = []
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d], x[d], ltype=ltype) * model.vaes[d].llik_scaling
            loc_lpx_z.append(lpx_z)
            if d == m:
                lpx_zs[m].append(lpx_z)
        elbo = (torch.stack(loc_lpx_z).sum(0) - kld.sum(-1).sum())
        elbos.append(elbo)
    return -torch.stack(elbos).sum(), torch.stack(klds).mean(0).sum(), -torch.stack(lpx_zs[0]).sum() / model.vaes[0].llik_scaling, -torch.stack(lpx_zs[1]).sum()
