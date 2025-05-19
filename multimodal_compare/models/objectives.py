# objectives of choice
import torch
import torch.distributions as dist
from utils import kl_divergence, is_multidata, log_mean_exp, log_joint, get_all_pairs, log_batch_marginal, softclip
from torch.autograd import Variable
from utils import kl_divergence, is_multidata, log_mean_exp, log_joint, get_all_pairs, log_batch_marginal, softclip
from numpy import prod
import numpy as np
from models.nn_modules import VGG19
import numpy as np
import copy


class BaseObjective():
    """
    Base objective class shared for all loss functions
    """
    def __init__(self):
        self.ltype = None
        self.beta = 1

    def set_ltype(self, ltype):
        """
        Checks the objective setup through a set of asserts
        """
        self.ltype = ltype
        assert hasattr(ReconLoss, self.ltype), "Loss function {} is not implemented. Choose from: {}".format(self.ltype,
                                            [func for func in dir(ReconLoss) if callable(getattr(ReconLoss, func))])

    def recon_loss_fn(self, output, target, K=1):
        """
        Calculate reconstruction loss

        :param output: Output data, torch.dist or list
        :type output:  torch.tensor
        :param target: Target data
        :type target: torch.tensor
        :param K: K samples from posterior distribution
        :type K: int
        :return: computed loss
        :rtype: torch.tensor
        """
        if target["masks"] is not None:
            output.loc = output.loc[:, :target["masks"].shape[1]]
            output.scale = output.loc[:, :target["masks"].shape[1]]
        # if target["data"].shape[1:] == torch.Size([3,64,64]):
        #     return torch.zeros((target["data"].shape[0],64)).cuda()
        target = target["data"]
        output, target = self.reshape_for_loss(output, target, K)
        bs = target.shape[0]
        loss = getattr(ReconLoss, self.ltype)(output, target, bs)
        return -loss

    def elbo(self, lpx_z, kld, beta=1):
        """
        The most general elbo function

        :param lpx_z: reconstruction loss(es)
        :type lpx_z: torch.tensor
        :param kld: KL divergence(s)
        :type kld: torch.tensor
        :param beta: disentangling factor
        :type beta: torch.float
        :return: ELBO loss
        :rtype: torch.tensor
        """
        return -(lpx_z.sum(-1) - beta * kld.sum()).sum()

    def iwae(self, lp_z, lpx_z, lqz_x):
        """
        The most general iwae function.

        :param lp_z: log probability of latent samples coming from the prior
        :type lp_z: torch.tensor
        :param lpx_z: reconstruction loss(es)
        :type lpx_z: torch.tensor
        :param lqz_x: log probability of latent samples coming from the learned posterior
        :type lqz_x: torch.tensor
        :return: IWAE loss
        :rtype: torch.tensor
        """
        lw = lp_z + lpx_z.sum(-1).mean(-1) - lqz_x
        return -log_mean_exp(lw).sum()

    @staticmethod
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

    def reshape_for_loss(self, output, target, K=1):
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
        if self.ltype != "category_ce":
            target = target.repeat(K, *([1]*(len(target.shape)-1))).reshape(*output.loc.shape)
        else:
            target = target.repeat(K, *([1] * (len(target.shape) - 1))).reshape(*output.loc.shape)
            if len(target.shape) == 2 and target.shape[-1] < 2:
                target = torch.nn.functional.one_hot(target)
        return output, target

    @staticmethod
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

    def calc_kld(self, dist1, dist2, cats=None):
        """
        Calculate KL divergence between two distributions
        :param dist1: distribution 1
        :type dist1: torch.dist
        :param dist2: distribution 2
        :type dist2: torch.dist
        :param cats: nmber of categories in case of gumbel softmax
        :type cats: int
        :return: KL divergence
        :rtype: torch.tensor
        """
        if not cats:
            return kl_divergence(dist1, dist2)
        else:
            N = int(dist1.param_shape[1] / cats)
            kld = self.calc_kld(dist1, dist.Categorical(
                torch.full((dist1.param_shape[0] * N, cats), 1.0 / cats).view(dist1.param_shape[0], -1).cuda()))
            return kld

    def calc_klds(self, latent_dists, model):
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
            klds.append(self.calc_kld(d, model.pz(*[x.cuda() for x in model.pz_params])))
        return klds

    def weighted_group_kld(self, latent_dists, model, weights):
        """
        Calculated the weighted group KL-divergence.

        :param latent_dists: list of the two distributions
        :type latent_dists: list
        :param model: model object
        :type model: object
        :param weights: tensor with weights for each distribution
        :type weights: torch.Tensor
        :return: group divergence, list of klds
        :rtype: tuple
        """
        klds = []
        for d in latent_dists:
            klds.append(self.calc_kld(d, model.pz(*[x.cuda() for x in model.pz_params])))
        group_div = torch.stack(klds).sum(-1).mean(1) * weights
        return group_div.sum(), klds


class UnimodalObjective(BaseObjective):
    """
    Common class for unimodal objectives (used in unimodal VAEs only)
    """
    def __init__(self, obj:str,beta=1):
        super().__init__()
        self.beta = beta
        self.objective = None
        self.obj_name = obj

    def calculate_loss(self, px_z, target, qz_x, prior_dist, pz_params, zs, K=1):
        """
        Calculates the loss using self.objective

        :param px_z: decoder distribution
        :type px_z: torch.distributions
        :param target: ground truth
        :type target: torch.tensor
        :param qz_x: posterior distribution
        :type qz_x: torch.distribution
        :param prior_dist: model's prior
        :type prior_dist: torch.distribution
        :param zs: latent samples
        :type zs: torch.tensor
        :param K: how many samples were drawn from the posterior
        :type K: int
        :return: calculated losses
        :rtype: dict
        """
        assert hasattr(self, self.obj_name), "Objective {} is not implemented in unimodal scenario".format(self.obj_name)
        self.objective = getattr(self, self.obj_name)
        data = {"px_z":px_z, "target":target, "qz_x": qz_x, "prior_dist":prior_dist, "zs":zs, "K": K, "pz_params":pz_params}
        output = self.objective(data)
        assert isinstance(output, dict), "Objective function must return a dictionary"
        return output


    def elbo(self, data):
        """
        Computes unimodal ELBO E_{p(x)}[ELBO]

        :param data: dict with the keys: px_z, target, qz_x, prior_dist, K
        :type dict: dict
        :return: dict with loss, kl divergence, reconstruction loss and kld
        :rtype: dict
        """
        lpx_z = self.recon_loss_fn(data["px_z"], data["target"], data["K"])
        kld = self.calc_kld(data["qz_x"], data["prior_dist"](*data["pz_params"]))
        loss = super().elbo(lpx_z, kld, self.beta)
        out = {"loss": loss, "kld": kld, "reconstruction_loss": lpx_z}
        return out

    def elbo_gumbel(self, data):
        """
        Computes unimodal ELBO E_{p(x)}[ELBO]

        :param data: dict with the keys: px_z, target, qz_x, prior_dist, K
        :type dict: dict
        :return: dict with loss, kl divergence, reconstruction loss and kld
        :rtype: dict
        """
        lpx_z = self.recon_loss_fn(data["px_z"], data["target"], data["K"])
        K = data["target"]["n_cats"]
        kld = self.calc_kld(data["qz_x"], data["prior_dist"], K)
        loss = super().elbo(lpx_z, kld, self.beta)
        out = {"loss": loss, "kld": kld, "reconstruction_loss": lpx_z}
        return out

    def iwae(self, data):
        """
        Computes an importance-weighted ELBO estimate for log p_\theta(x) Source: https://github.com/iffsid/mmvae

        :param data: dict with the keys: px_z, target, qz_x, zs, K
        :type dict: dict
        :return: dict with loss, reconstruction loss and kld
        :rtype: dict
        """
        lpx_z = self.recon_loss_fn(data["px_z"], data["target"], data["K"])
        lqz_x = data["qz_x"].log_prob(data["zs"]).sum(-1)
        lp_z = data["prior_dist"](*data["_pz_params"]).log_prob(data["zs"]).sum(-1)
        loss = super().iwae(lp_z, lpx_z, lqz_x)
        out = {"loss": loss, "reconstruction_loss": lpx_z, "kld": None}
        return out

    def dreg(self, data):
        """DREG estimate for log p_\theta(x) -- fully vectorised. Source: https://github.com/iffsid/mmvae

        :param data: dict with the keys: px_z, target, qz_x, zs, K, prior_dist
        :type dict: dict
        :return: dict with loss, reconstruction loss and kld
        :rtype: dict
        """
        lpz = data["prior_dist"].log_prob(data["zs"]).sum(-1)
        lpx_z = self.recon_loss_fn(data["px_z"], data["target"], K=data["K"]).view(
            *data["px_z"].batch_shape[:1], -1)
        lqz_x = data["qz_x"].log_prob(data["zs"]).sum(-1)
        lw = lpz + lpx_z.sum(-1) - lqz_x
        out = {"loss":lw, "reconstruction_loss": lpx_z, "kld":None}
        return out


class MultimodalObjective(BaseObjective):
    """
    Common class for multimodal objectives
    """
    def __init__(self, obj:str, beta=1):
        super().__init__()
        assert hasattr(self, obj), "Objective {} is not implemented in multimodal scenario".format(obj)
        self.beta = beta
        self.obj_name = obj
        self.objective = getattr(self, obj)

    def calculate_loss(self, data):
        """
        Calculates the loss using self.objective

        :param px_z: dictionary with the required data for loss calculation
        :type px_z: dict
        :return: calculated losses
        :rtype: dict
        """
        assert self.ltype is not None, "loss type is not set, please call set_ltype first"
        output = self.objective(data)
        assert isinstance(output, dict), "Objective function must return a dictionary"
        return output

    def elbo(self, data):
        """
        Computes multimodal ELBO E_{p(x)}[ELBO]

        :param data: dict with the keys: lpx_z (recon losses) and kld (kl divergences)
        :type dict: dict
        :return: dict with loss, kl divergence, reconstruction loss and kld
        :rtype: dict
        """
        loss = super().elbo(data["lpx_z"], data["kld"], self.beta)
        return {"loss":loss, "reconstruction_loss": data["lpx_z"], "kld": data["kld"]}

    def iwae(self, data):
        """
        Computes multimodal IWAE

        :param data: dict with the keys: lpx_z (recon losses) and kld (kl divergences)
        :type dict: dict
        :return: dict with loss, kl divergence, reconstruction loss and kld
        :rtype: dict
        """
        lws = []
        for r, qz_x in enumerate(data["qz_x"]):
            lpz = data["pz"](*data["pz_params"].cuda()).log_prob(data["zs"][r]["latents"]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(data["zs"][r]["latents"]).sum(-1) for qz_x in data["qz_x"]]))
            lpx_z = torch.stack(data["lpx_z"][r]).sum(0)
            lw = lpz + lpx_z.reshape(*lpz.shape) - self.beta * lqz_x
            lws.append(lw)
        loss = -log_mean_exp(torch.cat(lws)).sum()
        return {"loss": loss, "kld": torch.tensor(0), "reconstruction_loss": torch.stack([torch.stack(x)for x in data["lpx_z"]])}

    def _m_dreg_looser(self, lpx_zs, pz, pz_params, zss, qz_xs, K=1):
        """DREG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised;
        Source: https://github.com/iffsid/mmvae
        This version is the looser bound---with the average over modalities outside the log
        """
        lws = []
        for r, zs in enumerate(zss):
            lpz = pz(*pz_params).log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs if qz_x_ is not None]))
            lpx_z = torch.stack(lpx_zs[r]).sum(0) if isinstance(lpx_zs[r], list) else lpx_zs[r]
            lw = lpz.sum(-1) + lpx_z - lqz_x.sum(-1)
            lws.append(lw)
        return torch.stack(lws), torch.stack(zss)

    def dreg(self, data):
        """Computes dreg estimate for log p_\theta(x) for multi-modal vae;
        Source: https://github.com/iffsid/mmvae
        This version is the looser bound---with the average over modalities outside the log
        """
        lw, zss = self._m_dreg_looser(data["lpx_z"], data["pz"], [x.cuda() for x in data["pz_params"]], [z["latents"] for z in data["zs"]], data["qz_x"], K=data["K"])
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if zss.requires_grad:
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        rec_loss = torch.stack([torch.stack(x) for x in data["lpx_z"]]) if isinstance(data["lpx_z"], list) else(data["lpx_z"])
        return {"loss": -(grad_wt * lw).mean(0).sum(), "kld": torch.tensor(0),
                 "reconstruction_loss": rec_loss}

class ReconLoss():
    """ Class that stores reconstruction loss functions """
    @staticmethod
    def bce(output, target, bs):
        """
        Binary Cross-Entropy loss

        :param output: model output distribution
        :type output: torch.distributions
        :param target: ground truth tensor
        :type target: torch.tensor
        :param bs: batch size
        :type bs: int
        :return: calculated loss
        :rtype: torch.Tensor.float
        """
        return torch.nn.functional.binary_cross_entropy(output.loc.cpu(), target.float().cpu().detach(),
                                                        reduction="none").cuda().reshape(bs, -1)

    @staticmethod
    def lprob(output, target, bs):
        """
        Log-likelihood loss

        :param output: model output distribution
        :type output: torch.distributions
        :param target: ground truth tensor
        :type target: torch.tensor
        :param bs: batch size
        :type bs: int
        :return: calculated loss
        :rtype: torch.Tensor.float
        """
        out = output.log_prob(target.cuda()).view(*target.shape[:1], -1).double().reshape(bs, -1)
        out[torch.isnan(out)] = 0
        return -out

    @staticmethod
    def l1(output, target, bs):
        """
        L1 loss

        :param output: model output distribution
        :type output: torch.distributions
        :param target: ground truth tensor
        :type target: torch.tensor
        :param bs: batch size
        :type bs: int
        :return: calculated loss
        :rtype: torch.Tensor.float
        """
        l = torch.nn.L1Loss(reduction="none")
        return l(output.loc.cpu(), target.float().cpu().detach()).reshape(bs, -1)

    @staticmethod
    def mse(output, target, bs):
        """
        Mean squared error (squared L2 norm) loss

        :param output: model output distribution
        :type output: torch.distributions
        :param target: ground truth tensor
        :type target: torch.tensor
        :param bs: batch size
        :type bs: int
        :return: calculated loss
        :rtype: torch.Tensor.float
        """
        l = torch.nn.MSELoss(reduction="none")
        return l(output.loc.cuda(), target.float().cuda().detach()).reshape(bs, -1)

    @staticmethod
    def feature_loss(output, target, bs):
        """
        Suitable for images only
        Feature extractor loss taken from https://github.com/LukeDitria/CNN-VAE/blob/master/

        :param output: model output distribution
        :type output: torch.distributions
        :param target: ground truth tensor
        :type target: torch.tensor
        :param bs: batch size
        :type bs: int
        :return: calculated loss
        :rtype: torch.Tensor.float
        """
        if target.shape[-1] == 3:
            target = target.transpose(1,3)
            #output.loc = output.loc.transpose(1,3)
        feature_extractor = VGG19().to("cuda")
        l = torch.nn.MSELoss(reduction="none")
        feature_loss = feature_extractor(torch.cat((output.loc.cuda(), target.float().cuda().detach()), 0))
        loss = l(output.loc.cuda(), target.float().cuda().detach()) + feature_loss
        ls = loss.sum(-1).sum(-1).reshape(bs, -1) if len (loss.shape) > 3 else loss.sum(-1).reshape(bs, -1)
        return ls

    @staticmethod
    def category_ce(output, target, bs):
        """
        Categorical Cross-Entropy loss (for classification problems such as text)

        :param output: model output distribution
        :type output: torch.distributions
        :param target: ground truth tensor
        :type target: torch.tensor
        :param bs: batch size
        :type bs: int
        :return: calculated loss
        :rtype: torch.Tensor.float
        """
        l = torch.nn.CrossEntropyLoss(reduction="none")
        return l(output.loc.cuda(), target.float().cuda().detach()).reshape(bs, -1)

    @staticmethod
    def optimal_sigma(output, target, bs):
        """Calculate Gaussian NLL with optimal sigma as in Sigma VAE https://github.com/orybkin/sigma-vae-pytorch"""
        log_sigma = ((target.float().cuda().detach().cpu() - output.loc.cpu()) ** 2).mean(list(range(len(output.loc.shape))), keepdim=True).sqrt().log()
        while not log_sigma.shape == torch.Size([]):  # unwrap a nested tensor
            log_sigma = log_sigma[0]
        log_sigma = softclip(log_sigma, -6)
        return (torch.pow((target.float().cuda().detach() - output.loc) / log_sigma.exp(), 2).clone().detach() + log_sigma + 0.5 * np.log(2 * np.pi)).reshape(bs, -1)

