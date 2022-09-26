from torch import optim, nn
import torch
import pytorch_lightning as pl
import models
from models.vae import VAE
import os
from utils import make_kl_df, unpack_vae_outputs, data_to_device
from models import objectives
from models.config_cls import Config
from models.mmvae_base import TorchMMVAE
from visualization import plot_kls_df
from visualization import t_sne


class MultimodalVAE(pl.LightningModule):
    """
    Multimodal VAE trainer common for all architectures. Configures, trains and tests the model.

    :param feature_dims: dictionary with feature dimensions of raining data
    :type feature_dims: dict
    :param cfg: instance of Config class
    :type cfg: Config
    """

    def __init__(self, cfg, feature_dims: dict):
        super().__init__()
        self.config = cfg
        self.optimizer = None
        self.objective = None
        self.model = TorchMMVAE()
        self.feature_dims = feature_dims
        self.get_model()
        self.get_objective()

    def get_objective(self):
        """
        Find the selected objective
        """
        self.objective = getattr(objectives, ('multimodal_' if hasattr(self.model, 'vaes') else '')
                                 + ("_".join((self.config.obj, self.config.mixing)) if hasattr(self.model, 'vaes')
                                    else self.config.obj))

    def configure_optimizers(self):
        """
        Sets up the optimizer specified in the config
        """
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=float(self.config.lr), amsgrad=True)
        elif self.config.optimizer.lower() == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(self.model.parameters(), lr=float(self.config.lr), eps=1e-16,
                                       betas=(0.9, 0.999),
                                       weight_decouple=True, rectify=False, print_change_log=False)
        else:
            raise NotImplementedError
        return self.optimizer

    def get_model(self):
        """
        Sets up the model according to the config file
        """
        vaes = {}
        for i, m in enumerate(self.config.mods):
                vaes["mod_{}".format(i + 1)] = VAE(m["encoder"], m["decoder"], self.feature_dims[m["mod_type"]],
                                                   self.config.n_latents)
        if len(self.config.mods) > 1:
            vaes = nn.ModuleDict(vaes)
            self.model = getattr(models, self.config.mixing.lower())(vaes)
            assert isinstance(self.model, TorchMMVAE)
        else:
            self.model = vaes["mod_1"]  # unimodal VAE scenario

        if self.config.pre_trained:
            print('Loading model {} from {}'.format(self.model.modelName, self.config.pre_trained))
            self.model = self.load_from_checkpoint(self.config.pre_trained, cfg=self.config)
        return self.model

    def training_step(self, train_batch, batch_idx):
        """
        Iterates over the train loader
        """
        loss, kld, partial_l = self.objective(self.model, train_batch, ltype=self.config.loss)
        self.log('train_loss', loss, batch_size=self.config.batch_size)
        self.log('train_kld', kld, batch_size=self.config.batch_size)
        for i, p_l in enumerate(partial_l):
            self.log("Mod_{}_TrainLoss".format(i), p_l, batch_size=self.config.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Iterates over the test loader
        """
        loss, kld, partial_l = self.objective(self.model, val_batch, ltype=self.config.loss)
        self.log('val_loss', loss, batch_size=self.config.batch_size)
        self.log('val_kld', kld, batch_size=self.config.batch_size)
        for i, p_l in enumerate(partial_l):
            self.log("Mod_{}_ValLoss".format(i), p_l, batch_size=self.config.batch_size)
        return loss

    def validation_epoch_end(self, outputs):
        """
        Save visualizations

        :param outputs: Loss that comes from validation_step
        :type outputs: torch.tensor
        """
        if (self.trainer.current_epoch + 1) % self.config.viz_freq == 0:
            self.analyse_data()

    def analyse_data(self, data=None, labels=None, num_samples=250, path_label=""):
        """
        Encodes data and plots T-SNE.
        :param data: test data
        :type data: torch.tensor
        :param labels: labels for the data for labelled T-SNE (optional) - list of strings
        :type labels: list
        :param num_samples: number of samples to use for visualization
        :type num_samples: int
        :param path_label: label under which to save the visualizations
        :type path_label: str
        :return: returns K latent samples for each input
        :rtype: list
        """
        data = next(iter(self.trainer.datamodule.predict_dataloader(num_samples))) if not data else data
        for key in data.keys():
            data[key] = {k: v.to(device=self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in data[key].items()}
        output = self.model.forward(data)
        qz_xs, zss, _ = unpack_vae_outputs(output)
        pz = self.model.pz(*[x for x in self.model.pz_params()])
        zss_sampled = [pz.sample(torch.Size([1, num_samples])).view(-1, pz.batch_shape[-1]),
               *[zs["latents"].view(-1, zs["latents"].size(-1)) for zs in zss]]
        kl_df = make_kl_df(qz_xs, pz)
        if path_label == "" and not self.config.eval_only:
                path_label = "_e_{}".format(self.trainer.current_epoch)
        plot_kls_df(kl_df, os.path.join(self.config.mPath, 'visuals/kl_distance{}.png'.format(path_label)))
        t_sne([x for x in zss_sampled[1:]], os.path.join(self.config.mPath, 'visuals/t_sne{}.png'.format(path_label)), labels)
