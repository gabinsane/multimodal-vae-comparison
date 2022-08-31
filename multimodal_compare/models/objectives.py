# objectives of choice
import torch
import torch.distributions as dist
from utils import kl_divergence, is_multidata, log_mean_exp, log_joint, get_all_pairs, log_batch_marginal
from torch.autograd import Variable
from numpy import prod
import copy


def compute_microbatch_split(x, K):
    """
    Checks if batch needs to be broken down further to fit in memory.
    :param x: input data
    :type x: torch.tensor
    :param K: K samples will be made from each distribution
    :type K: int
    :return: microbatch split
    :rtype: torch.tensor
    """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def reshape_for_loss(output, target, ltype, K=1):
    """
    Reshapes output and target to calculate reconstruction loss

    :param output: output likelihood
    :type output: torch.dist
    :param target: target likelihood
    :type target: torch.dist
    :param ltype: reconstruction loss
    :type ltype: str
    :param K: K samples from posterior distribution
    :type K: int
    :return: reshaped data
    :rtype: tuple(torch.tensor, torch.tensor, str)
    """
    target = torch.stack(target).float() if isinstance(target, list) else target
    if target.shape[-2] != output.loc.shape[-2]:
        target = torch.nn.functional.pad(target, pad=(0, 0, 0, output.loc.shape[1] - target.shape[1]), mode='constant',
                                         value=0)
    target = target.repeat(K, 1, 1, 1).reshape(*output.loc.shape)
    return output, target, ltype


def loss_fn(output, target, ltype, categorical=False, K=1):
    """
    Calculate reconstruction loss

    :param output: Output data, torch.dist or list
    :type output:  torch.tensor
    :param target: Target data
    :type target: torch.tensor
    :param categorical: whether the data is categorical (calculates classification loss)
    :type categorical: bool
    :param K: K samples from posterior distribution
    :type K: int
    :return: computed loss
    :rtype: torch.tensor
    """
    output, target, ltype = reshape_for_loss(output, target, ltype, K)
    bs = target.shape[0]
    if ltype in ["bce"] and categorical:
        ltype = "category" # use category cross entropy instead
    if ltype == "bce":
        loss = torch.nn.functional.binary_cross_entropy(output.loc.cpu(), target.float().cpu().detach(), reduction="none").cuda().reshape(bs, -1)
    elif ltype == "lprob":
        loss = -output.log_prob(target.cuda()).view(*target.shape[:1], -1).double().reshape(bs, -1)
    elif ltype == "l1":
        l = torch.nn.L1Loss(reduction="none")
        loss = l(output.loc.cpu(), target.float().cpu().detach()).reshape(bs, -1)
    elif ltype == "mse":
        l = torch.nn.MSELoss(reduction="none")
        loss = l(output.loc.cuda(), target.float().cuda().detach()).reshape(bs, -1)
    elif ltype == "category":
        l = torch.nn.CrossEntropyLoss(reduction="none")
        loss = l(output.loc.cuda(), target.float().cuda().detach()).reshape(bs, -1)
    return -loss


def normalize(target, data=None):
    """
    Normalize data between 0 and 1

    :param target: target data
    :type target: torch.tensor
    :param data: output data (optional)
    :type data: torch.tensor
    :return: normalized data
    :rtype: list
    """
    t_size= target.size()
    maxv, minv = torch.max(target.reshape(-1)), torch.min(target.view(-1))
    output = [torch.div(torch.add(target.reshape(-1), torch.abs(minv)), (maxv-minv)).reshape(t_size)]
    if data is not None:
        d_size = data.size()
        data_norm = torch.clamp(torch.div(torch.add(data.reshape(-1), torch.abs(minv)), (maxv-minv)), min=0, max=1)
        output.append(data_norm.reshape(d_size))
    return output


def calc_klds(latent_dists, model):
    """
    Calculated th KL-divergence between the distribution and posterior dist.

    :param latent_dists: list of the two distributions
    :type latent_dists: list
    :param model: model object
    :type model: object
    :return: list of klds
    :rtype: list
    """
    klds = []
    for d in latent_dists:
        klds.append(kl_divergence(d, model.pz(*model._pz_params)))
    return klds


def elbo(model, x, ltype="lprob"):
    """
    Computes unimodal ELBO E_{p(x)}[ELBO]

    :param model: model object
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    output = model(x["mod_1"])
    qz_x = output["mod_1"].encoder_dists
    px_z = output["mod_1"].decoder_dists
    lpx_z = loss_fn(px_z, x["mod_1"]["data"], ltype=ltype, categorical= x["mod_1"]["categorical"])
    kld = kl_divergence(qz_x, model.pz(*model._pz_params))
    return -(lpx_z.sum(-1) - kld.sum()).sum(), kld.sum(), [-lpx_z.sum()]


def multimodal_elbo_moe(model, x, ltype="lprob"):
    """Computes ELBO for MoE VAE as in https://github.com/iffsid/mmvae

    :param model: model object
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    output_dict = model(x)
    qz_xs, px_zs, zss =[], [[None for _ in range(len(x.keys()))] for _ in range(len(x.keys()))], []
    for ind, mod in enumerate(output_dict.keys()):
        qz_xs.append(output_dict[mod].encoder_dists)
        zss.append(output_dict[mod].latent_samples["latents"])
        px_zs[ind][ind] = output_dict[mod].decoder_dists[0]
        pos = 0 if ind == 1 else 1
        px_zs[ind][pos]  = output_dict[mod].decoder_dists[1]
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.vaes["mod_{}".format(r+1)]._pz_params.cuda()))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = loss_fn(px_zs[d][d], x["mod_{}".format(d+1)]["data"], ltype=ltype, categorical= x["mod_{}".format(d+1)]["categorical"]).view(*px_zs[d][d].batch_shape[:1], -1)
            lpx_z = (lpx_z * model.vaes["mod_{}".format(d+1)].llik_scaling).sum(-1)
            if d == r:
                  lwt = torch.tensor(0.0).cuda()
            else:
                  zs = zss[d].detach()
                  qz_x.log_prob(zs)[torch.isnan(qz_x.log_prob(zs))] = 0
                  lwt = (qz_x.log_prob(zs)- qz_xs[d].log_prob(zs).detach()).sum(-1)[0][0]
            lpx_zs.append((lwt.exp() * lpx_z))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    individual_losses = [-m.sum() / model.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs[0::len(x)+1])]
    return -obj.sum(), torch.stack(klds).mean(0).sum(), individual_losses


def multimodal_elbo_mopoe(model, x, ltype="lprob", beta=5):
    """Computes GENERALIZED MULTIMODAL ELBO https://arxiv.org/pdf/2105.02470.pdf

    :param model: model object
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    qz_xs, px_zs, zss, single_latents = model(x)
    lpx_zs, klds = [], []
    uni_mus, uni_logvars = list(single_latents[0][:-1].squeeze(1)), list(single_latents[1][:-1].squeeze(1))
    uni_dists = [dist.Normal(*[mu, logvar]) for mu, logvar in zip(uni_mus, uni_logvars)]
    for r, px_z in enumerate(px_zs):
        tag = "mod_{}".format(r+1)
        lpx_z = loss_fn(px_z, x[tag]["data"], ltype=ltype, categorical=x[tag]["categorical"]).cuda() * model.vaes[r].llik_scaling
        lpx_zs.append(lpx_z.sum(-1))
    rec_loss = torch.stack(lpx_zs).sum()/len(lpx_zs)
    rec_loss = Variable(rec_loss, requires_grad=True)
    group_divergence = kl_divergence(qz_xs, model.pz(*model._pz_params))
    kld_mods = calc_klds(uni_dists, model)
    kld_weighted = (torch.stack(kld_mods).sum(0) + group_divergence).sum()
    obj = rec_loss #- beta * kld_weighted
    individual_losses = [-m.sum() / model.vaes[idx].llik_scaling for idx, m in enumerate(lpx_zs)]
    return -obj.sum(), kld_weighted, individual_losses


def multimodal_elbo_poe(model, x,  ltype="lprob"):
    """Subsampled ELBO with the POE approach as used in https://github.com/mhw32/multimodal-vae-public

    :param model: model object
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    lpx_zs, klds, elbos = [[] for _ in range(len(x))], [], []
    for m in range(len(x.keys()) + 1):
        mods = copy.deepcopy(x)
        for d in mods.keys():
            mods[d]["data"] = None
            mods[d]["masks"] = None
        if m == len(x.keys()):
            mods = x
        else:
            mods["mod_{}".format(m+1)] = x["mod_{}".format(m+1)]
        output_dic = model(mods)
        kld = kl_divergence(output_dic["mod_1"].encoder_dists, model.pz(*model.vaes["mod_1"]._pz_params))
        klds.append(kld.sum(-1))
        loc_lpx_z = []
        for mod in output_dic.keys():
            px_z = output_dic[mod].decoder_dists[0]
            lpx_z = (loss_fn(px_z, x[mod]["data"], ltype=ltype, categorical=x[mod]["categorical"]) * model.vaes[d].llik_scaling).sum(-1)
            loc_lpx_z.append(lpx_z)
            if mod == "mod_{}".format(m+1):
                lpx_zs[m].append(lpx_z)
        elbo = (torch.stack(loc_lpx_z).sum(0) - kld.sum(-1).sum())
        elbos.append(elbo)
    individual_losses = [-torch.stack(m).sum() / model.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs)]
    return -torch.stack(elbos).sum(), torch.stack(klds).mean(0).sum(), individual_losses

def iwae(model, x,  ltype="lprob", beta=1, K=20):
    """
    Computes an importance-weighted ELBO estimate for log p_\theta(x) Source: https://github.com/iffsid/mmvae

    :param model: VAE class
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :param beta: beta parameter
    :type beta: int
    :param K: number of samples from posterior
    :type K: int
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    output = model(x["mod_1"], K)
    qz_x = output["mod_1"].encoder_dists
    px_z = output["mod_1"].decoder_dists
    zs = output["mod_1"].latent_samples
    lpz = model.pz(*model._pz_params).log_prob(zs).sum(-1)
    lpx_z = loss_fn(px_z, x["mod_1"]["data"], ltype=ltype, categorical=x["mod_1"]["categorical"], K=K)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1).mean(-1) - lqz_x
    return -log_mean_exp(lw).sum(), torch.zeros(1), [torch.zeros(1)]


def multimodal_iwae_moe(model, x, K=1, ltype="lprob"):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised  Source: https://github.com/iffsid/mmvae

    :param model: multimodal VAE class
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :param K: number of samples from posterior
    :type K: int
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model._pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [loss_fn(px_z, x["mod_{}".format(d + 1)]["data"], ltype=ltype, categorical=x["mod_{}".format(d + 1)]["categorical"], K=K).view(*px_z.batch_shape[:1], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return -torch.cat(lws).sum(), torch.zeros(1), [torch.zeros(1)] * len(x)


def dreg(model, x, K, ltype="lprob"):
    """DREG estimate for log p_\theta(x) -- fully vectorised. Source: https://github.com/iffsid/mmvae

    :param model: VAE class
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :param K: number of samples from posterior
    :type K: int
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    output = model(x["mod_1"], K)
    qz_x = output["mod_1"].encoder_dists
    px_z = output["mod_1"].decoder_dists
    zs = output["mod_1"].latent_samples
    lpz = model.pz(*model._pz_params).log_prob(zs).sum(-1)
    lpx_z = loss_fn(px_z, x["mod_1"]["data"], ltype=ltype, categorical=x["mod_1"]["categorical"], K=K).view(*px_z.batch_shape[:1], -1) * model.llik_scaling
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs, torch.zeros(1), [torch.zeros(1)]


def _m_dreg_looser(model, x, ltype, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae - fully vectorised
    This version is the looser bound with the average over modalities outside the log
    Source: https://github.com/iffsid/mmvae

    :param model: multimodal VAE class
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :param K: number of samples from posterior
    :type K: int
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model._pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [loss_fn(px_z, x["mod_{}".format(d + 1)]["data"], ltype=ltype, categorical=x["mod_{}".format(d + 1)]["categorical"], K=K).view(*px_z.batch_shape[:1], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        if lpx_z.shape[0] > lpz.shape[0]:
            lpx_z = lpx_z.reshape(*lpz.shape).sum(-1)
        lw = lpz.sum(-1) + lpx_z - lqz_x.sum(-1)
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss)


def multimodal_dreg_looser_moe(model, x, K=30, ltype="lprob"):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    Source: https://github.com/iffsid/mmvae

    :param model: multimodal VAE class
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :param K: number of samples from posterior
    :type K: int
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    lw, zss = _m_dreg_looser(model, x, ltype, K)
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return -(grad_wt * lw).mean(0).sum(), torch.zeros(1), [torch.zeros(1)] * len(x)


def multimodal_elbo_dmvae(model, x, K=1, ltype="lprob"):
    """
    Objective for the DMVAE model. Source: https://github.com/seqam-lab/
    
    :param model: VAE class
    :type model: object
    :param x: input batch
    :type x: torch.tensor
    :param ltype: reconstruction loss term
    :type ltype: str
    :param K: number of samples from posterior
    :type K: int
    :return: loss, kl divergence, reconstruction loss
    :rtype: tuple(torch.tensor, torch.tensor, list)
    """
    qz_xs, px_zs, zss = model(x)
    recons = []
    kls = []
    ind_losses = []
    for i in range(len(px_zs)):
        tag = "mod_{}".format(i + 1)
        for j in range(len(px_zs[i])):
            if j < len(px_zs[i])-1:
                recons.append(loss_fn(px_zs[i][j], x[tag]["data"], ltype=ltype, categorical=x[tag]["categorical"]).cuda() * model.vaes[i].llik_scaling.mean())
            else:
                recons.append(torch.tensor(0))
        for n in range(len(px_zs[i])-1):
            idxs = [2+n,4]
            log_pz = log_joint([px_zs[i][n], px_zs[n+1]], [zss[i], zss[idxs[n]]])
            log_q_zCx = log_joint([qz_xs[i], qz_xs[idxs[n]]], [zss[i], zss[idxs[n]]])
            log_qz, _, log_prod_qzi = log_batch_marginal([qz_xs[i], qz_xs[idxs[n]]])
            kl = ((log_q_zCx - log_qz) *(log_qz - log_prod_qzi)* (log_prod_qzi - log_pz)).mean()
            kls.append(kl)
    # cross sampling
    for i in get_all_pairs(px_zs):
        tag = "mod_{}".format(i + 1)
        recons.append(loss_fn(px_zs[i[0]][0], x[tag]["data"], ltype=ltype, categorical=x[tag]["categorical"]).cuda() * model.vaes[
            i].llik_scaling.mean())
        log_pz = log_joint([px_zs[i[0]][0], px_zs[i[0]][1]])
        log_q_zCx = log_joint([qz_xs[i[0]][0], qz_xs[i[0]][1]])
        log_qz, _, log_prod_qzi = log_batch_marginal([qz_xs[i[0]][0], qz_xs[i[0]][1]])
        kl =  ((log_q_zCx - log_qz) *(log_qz - log_prod_qzi)* (log_prod_qzi - log_pz)).mean()
        kls.append(kl)
    for rec, kl in zip(recons, kls):
        l = rec - kl
        ind_losses.append(l)
    loss = torch.tensor(ind_losses).sum()
    return -loss, torch.stack(kls).sum(), ind_losses

