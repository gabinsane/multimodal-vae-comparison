# objectives of choice
import torch
import torch.distributions as dist
from utils import log_mean_exp, is_multidata, kl_divergence
from torch.autograd import Variable

def loss_fn(output, target, ltype, mod_type=None):
    if len(target) == 2:
        target = target[0]
    target = torch.stack(target).float() if isinstance(target, list) else target
    if mod_type is not None and "transformer" in mod_type.lower():
        if ltype !="lprob":
            output.loc = output.loc[:,:target.shape[1]]
            output.scale = output.scale[:, :target.shape[1]]
        if "txt" in mod_type.lower() and ltype !="lprob":
            ltype = "category"
    else:
        target = target.reshape(*output.loc.shape)
    if ltype == "bce":
        assert torch.min(target.reshape(-1)) >= 0 and torch.max(target.reshape(-1)) <= 1, "Cannot use bce on data which is not normalised"
        loss = -torch.nn.functional.binary_cross_entropy(output.loc.squeeze().cpu(), target.float().cpu().detach(), reduction="sum").cuda()
    elif ltype == "lprob":
        if "transformer" in mod_type.lower() and output.loc.shape != target.shape:
            target = torch.nn.functional.pad(target, pad=(0,0,0,output.loc.shape[1] - target.shape[1]), mode='constant', value=0)
        loss = output.log_prob(target.cuda()).view(*target.shape[:2], -1).sum(-1).sum(-1).sum(-1).double()
    elif ltype == "l1":
        l = torch.nn.L1Loss(reduction="sum")
        loss = -l(output.loc.cpu(), target.float().cpu().detach())
    elif ltype == "mse":
        l = torch.nn.MSELoss(reduction="sum")
        loss = -l(output.loc.cuda(), target.float().cuda().detach())
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

def elbo(model, x, ltype="lprob"):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = loss_fn(px_z, x, ltype=ltype)
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return -(lpx_z.sum(-1) - kld.sum()).sum(), kld.sum(), [-lpx_z.sum()]

def multimodal_elbo_moe(model, x, ltype="lprob"):
    """Computes ELBO for MoE VAE"""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d][d], x[d], ltype=ltype, mod_type=model.vaes[d].dec_name).cuda() * model.vaes[d].llik_scaling
            if d == r:
                  lwt = torch.tensor(0.0).cuda()
            else:
                  zs = zss[d].detach()
                  qz_x.log_prob(zs)[torch.isnan(qz_x.log_prob(zs))] = 0
                  lwt = (qz_x.log_prob(zs)- qz_xs[d].log_prob(zs).detach()).sum(-1)[0][0]
            lpx_zs.append((lwt.exp() * lpx_z))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    individual_losses = [-m.sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs[0::len(x)+1])]
    return -obj.sum(), torch.stack(klds).mean(0).sum(), individual_losses


def multimodal_elbo_mopoe(model, x, ltype="lprob", beta=5):
    """Computes GENERALIZED MULTIMODAL ELBO https://arxiv.org/pdf/2105.02470.pdf """
    qz_xs, px_zs, zss, single_latents = model(x)
    lpx_zs, klds = [], []
    uni_mus, uni_logvars = list(single_latents[0][:-1].squeeze(1)), list(single_latents[1][:-1].squeeze(1))
    uni_dists = [dist.Normal(*[mu, logvar]) for mu, logvar in zip(uni_mus, uni_logvars)]
    for r, px_z in enumerate(px_zs):
        lpx_z = loss_fn(px_z, x[r], ltype=ltype, mod_type=model.vaes[r].dec_name).cuda() * model.vaes[r].llik_scaling
        lpx_zs.append(lpx_z)
    rec_loss = torch.tensor(lpx_zs).sum()/len(lpx_zs)
    rec_loss = Variable(rec_loss, requires_grad=True)
    group_divergence = kl_divergence(qz_xs, model.pz(*model.pz_params))
    kld_mods = calc_klds(uni_dists, model)
    kld_weighted = (torch.stack(kld_mods).sum(0) + group_divergence).sum()
    obj = rec_loss #- beta * kld_weighted
    individual_losses = [-m.sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs)]
    return -obj.sum(), kld_weighted, individual_losses


def multimodal_elbo_poe(model, x,  ltype="lprob"):
    lpx_zs, klds, elbos = [[] for _ in range(len(x))], [], []
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
            lpx_z = loss_fn(px_zs[d], x[d], ltype=ltype, mod_type=model.vaes[d].dec_name) * model.vaes[d].llik_scaling
            loc_lpx_z.append(lpx_z)
            if d == m:
                lpx_zs[m].append(lpx_z)
        elbo = (torch.stack(loc_lpx_z).sum(0) - kld.sum(-1).sum())
        elbos.append(elbo)
    individual_losses = [-torch.stack(m).sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs)]
    return -torch.stack(elbos).sum(), torch.stack(klds).mean(0).sum(), individual_losses

