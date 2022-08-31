from torch import optim, nn
import torch
import pytorch_lightning as pl
import models
from models.vae import VAE
import os
from utils import make_kl_df
from models import objectives
from models.config_cls import Config
from models.mmvae_base import TorchMMVAE, BaseVae
from visualization import plot_kls_df


class MultimodalVAE(pl.LightningModule):
    """
    Multimodal VAE trainer common for all architectures. Configures, trains and tests the model.

    :param cfg: instance of Config class
    :type cfg: Config
    """

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.optimizer = None
        self.objective = None
        self.model = TorchMMVAE()
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
                vaes["mod_{}".format(i + 1)] = VAE(m["encoder"], m["decoder"], m["feature_dim"], self.config.n_latents)
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

        assert isinstance(self.model, TorchMMVAE)
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
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.config.viz_freq == 0:
            self.visualize_latents()
        return loss

    def analyse_data(self, data=None, labels=None):
        """
        Encodes data and plots T-SNE.
        :param data: test data
        :type data: torch.tensor
        :param labels: labels for the data for labelled T-SNE (optional) - list of strings
        :type labels: list
        :return: returns K latent samples for each input
        :rtype: list
        """
        if not data:
            data = next(iter(self.trainer.datamodule.predict_dataloader(250)))
        with torch.no_grad():
            output = self.model.forward(data)
            qz_xs = [output[m].encoder_dists for m in output.keys()]
            zss = [output[m].latent_samples for m in output.keys()]
            pz = self.model.pz(*self.model.pz_params)
            zss_sampled = [pz.sample(torch.Size([1, len(data[0])])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
        kl_df = make_kl_df(qz_xs, pz)
        plot_kls_df(kl_df, os.path.join(self.config.mPath, 'visuals/kl_distance_{}.png'.format(self.trainer.current_epoch)))
        #t_sne([x.cpu() for x in zss_sampled[1:]], runPath, epoch, 1, labels)
