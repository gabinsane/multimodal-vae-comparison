import torch
from torch import optim
import pytorch_lightning as pl
import models
from models import objectives
from models.vae import VAE


class Trainer(pl.LightningModule):
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
        self.model = None
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

    def get_model(self):
        """
        Sets up the model according to the config file
        """
        vaes = []
        for i, m in enumerate(self.config.mods):
            vaes.append(VAE(m["encoder"], m["decoder"], m["feature_dim"], self.config.n_latents))
        self.model = getattr(models, self.config.mixing.lower())(*vaes) if len(vaes) > 1 else vaes[0]
        if self.config.pre_trained:
            print('Loading model {} from {}'.format(model.modelName, self.config.pre_trained))
            self.model.load_state_dict(torch.load(self.config.pre_trained + '/model.rar'))
            self.model._pz_params = model._pz_params

    def training_step(self, train_batch, batch_idx):
        """
        Iterates over the train loader
        """
        loss, kld, partial_l = self.objective(self.model, train_batch, ltype=self.config.loss)
        self.log('train_loss', loss)
        self.log('train_kld', kld)
        for i, p_l in enumerate(partial_l):
            self.log("Mod_{}_TrainLoss".format(i), p_l)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Iterates over the test loader
        """
        loss, kld, partial_l = self.objective(self.model, val_batch, ltype=self.config.loss)
        self.log('val_loss', loss)
        self.log('val_kld', kld)
        for i, p_l in enumerate(partial_l):
            self.log("Mod_{}_ValLoss".format(i), p_l)
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.config.viz_freq == 0:
            self.visualize_latents()
        return loss

    def visualize_latents(self):
        """
        Runs the model analysis, saves T-SNE and KL-Divergences
        """
        pass

    def make_reconstructions(self):
        """
        Make reconstructions and save into vis dir
        """

    def generate_traversals(self):
        """
        Make traversals and save into vis dir
        """