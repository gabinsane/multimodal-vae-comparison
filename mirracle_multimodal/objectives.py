# objectives of choice
import torch
from numpy import prod
import numpy as np
import torch.distributions as dist
from utils import log_mean_exp, is_multidata, kl_divergence
from torch.autograd import Variable

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

def loss_fn(output, target, ltype, mod_type=None):
    if len(target) == 2:
        target = target[0]
    target = torch.stack(target).float() if isinstance(target, list) else target
    if mod_type is not None and "transformer" in mod_type.lower():
        output.loc = output.loc[:,:target.shape[1]]
        output.scale = output.scale[:, :target.shape[1]]
        if "txt" in mod_type.lower():
            ltype = "category"
    else:
        target = target.reshape(*output.loc.shape)
    if ltype == "bce":
        output = output.loc
        assert torch.min(target.reshape(-1)) >= 0 and torch.max(target.reshape(-1)) <= 1, "Cannot use bce on data which is not normalised"
        loss = -torch.nn.functional.binary_cross_entropy(output.squeeze().cpu(), target.float().cpu().detach(), reduction="sum").cuda()
    elif ltype == "lprob":
        loss = output.log_prob(target.cuda()).view(*target.shape[:2], -1).sum(-1).sum(-1).sum(-1).double()
    elif ltype == "l1":
        l = torch.nn.L1Loss(reduction="sum")
        loss = -l(output.loc.cpu(), target.float().cpu().detach())
    elif ltype == "category":
        l = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = -l(output.loc.cuda(), target.float().cuda().detach())
    return loss


def normalize(target, data=None):
    t_size= target.size()
    maxv, minv = torch.max(target.reshape(-1)), torch.min(target.view(-1))
    output = [torch.div(torch.add(target.reshape(-1), torch.abs(minv)), (maxv-minv)).reshape(t_size)]
    if data is not None:
        d_size = data.size()
        data_norm = torch.clamp(torch.div(torch.add(data.reshape(-1), torch.abs(minv)), (maxv-minv)), min=0, max=1)
        output.append(data_norm.reshape(d_size))
    return output


def calc_klds(latent_dists, model):
    klds = []
    for d in latent_dists:
        klds.append(kl_divergence(d, model.pz(*model.pz_params)))
    return klds


def elbo(model, x, d_len, K=1, ltype="lprob"):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = loss_fn(px_z, x, ltype=ltype)/d_len
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))/d_len
    return -(lpx_z.sum(-1) - kld.sum()).sum(), kld.sum(), [-lpx_z.sum()]


def _iwae(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1).sum(-1) - lqz_x


def iwae(model, x, K=1, ltype=None):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return -log_mean_exp(lw).sum(), 0,[0]


def _dreg(model, x, K):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs


def dreg(model, x, K, regs=None):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw, zs = zip(*[_dreg(model, _x, K) for _x in x.split(S)])
    lw = torch.cat(lw, 1)  # concat on batch
    zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()

def m_elbo_moe(model, x, d_len, ltype="lprob"):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1)/d_len)
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d][d], x[d], ltype=ltype, mod_type=model.vaes[d].dec_name).cuda() * model.vaes[d].llik_scaling
            if d == r:
                  lwt = torch.tensor(0.0).cuda()
            else:
                  zs = zss[d].detach()
                  qz_x.log_prob(zs)[torch.isnan(qz_x.log_prob(zs))] = 0
                  lwt = (qz_x.log_prob(zs)- qz_xs[d].log_prob(zs).detach()).sum(-1)[0][0]
            lpx_zs.append((lwt.exp() * lpx_z)/d_len)
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    individual_losses = [-m.sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs[0::len(x)+1])]
    return -obj.sum(), torch.stack(klds).mean(0).sum(), individual_losses


def m_elbo_mopoe(model, x, d_len, ltype="lprob", beta=5):
    """Computes GENERALIZED MULTIMODAL ELBO https://arxiv.org/pdf/2105.02470.pdf """
    qz_xs, px_zs, zss, single_latents = model(x)
    lpx_zs, klds = [], []
    uni_mus, uni_logvars = list(single_latents[0][:-1].squeeze(1)), list(single_latents[1][:-1].squeeze(1))
    uni_dists = [dist.Normal(*[mu, logvar]) for mu, logvar in zip(uni_mus, uni_logvars)]
    for r, px_z in enumerate(px_zs):
        lpx_z = loss_fn(px_z, x[r], ltype=ltype, mod_type=model.vaes[r].dec_name).cuda() * model.vaes[r].llik_scaling
        lpx_zs.append(lpx_z/d_len)
    rec_loss = torch.tensor(lpx_zs).sum()
    group_divergence = kl_divergence(qz_xs, model.pz(*model.pz_params))
    kld_mods = calc_klds(uni_dists, model)
    kld_weighted = (torch.stack(kld_mods).sum(0) + group_divergence).sum()/d_len
    rec_loss = Variable(rec_loss, requires_grad=True)
    obj = rec_loss - beta * kld_weighted
    individual_losses = [-m.sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs)]
    return -obj.sum(), kld_weighted, individual_losses


def m_elbo_binding_moe(model, x, d_len, ltype="lprob"):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    bindings = []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        if r != len(qz_xs)-1:
            kld_b = kl_divergence(qz_x, qz_xs[r+1])
            bindings.append(kld_b)
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d][d], x[d], ltype=ltype).cuda() * model.vaes[d].llik_scaling
            if d == r:
                  lwt = torch.tensor(0.0).cuda()
            else:
                  zs = zss[d].detach()
                  qz_x.log_prob(zs)[torch.isnan(qz_x.log_prob(zs))] = 0
                  lwt = (qz_x.log_prob(zs)- qz_xs[d].log_prob(zs).detach()).sum(-1)[0][0]
            lpx_zs.append((lwt.exp() * lpx_z))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(bindings).sum(0).sum()) #torch.stack(klds).sum(0) -
    individual_losses = [-m.sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs[0::len(x)+1])]
    return -obj.sum(), torch.stack(bindings).sum(0).mean(0).sum(), individual_losses

def m_elbo_poe(model, x, d_len, ltype="lprob", ):
    lpx_zs, klds, elbos = [[] for _ in range(len(x))], [], []
    for m in range(len(x) + 1):
        mods = [None for _ in range(len(x))]
        if m == len(x):
            mods = x
        else:
            mods[m] = x[m]
        qz_x, px_zs, _ = model(mods)
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1)/d_len)
        loc_lpx_z = []
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d], x[d], ltype=ltype, mod_type=model.vaes[d].dec_name) * model.vaes[d].llik_scaling/d_len
            loc_lpx_z.append(lpx_z)
            if d == m:
                lpx_zs[m].append(lpx_z)
        elbo = (torch.stack(loc_lpx_z).sum(0) - kld.sum(-1).sum())
        elbos.append(elbo)
    individual_losses = [-torch.stack(m).sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs)]
    return -torch.stack(elbos).sum(), torch.stack(klds).mean(0).sum(), individual_losses


def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae(model, x, K=1, ltype=""):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return -log_mean_exp(lw).sum(), 0, -log_mean_exp(lw[:10]).sum(), -log_mean_exp(lw[10:]).sum()


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae_looser(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae_looser(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws), torch.cat(zss)


def m_dreg(model, x, K=1, ltype=""):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 1)  # concat on batch
    zss = torch.cat(zss, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return -(grad_wt * lw).sum(), 0, 0, 0

def _m_dreg_looser(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    lpx_zs = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lpx_zs.append(lpx_z)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss), lpx_zs


def m_dreg_looser(model, x, K=1, ltype=""):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss, lpx_zs = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    zss = torch.cat(zss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return -(grad_wt * lw).mean(0).sum(), 0, -lpx_zs[0][0].mean(0).sum(),  -lpx_zs[0][1].mean(0).sum()