from torch import optim, nn
import torch
import pytorch_lightning as pl
import models
from models.vae import VAE
import os, copy
from utils import make_kl_df, data_to_device, check_input_unpacked
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
        self.mod_names = self.get_mod_names()
        self.feature_dims = feature_dims
        self.get_model()

    def get_mod_names(self):
        mod_names = {}
        for i, m in enumerate(self.config.mods):
            mod_names["mod_{}".format(i+1)] = m["mod_type"]
        return mod_names

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
                                                   self.config.n_latents, m["recon_loss"], m["private_latents"], obj_fn=self.config.obj,
                                                   beta=self.config.beta, id_name="mod_{}".format(i + 1))
        if len(self.config.mods) > 1:
            vaes = nn.ModuleDict(vaes)
            obj_cfg = {"obj":self.config.obj, "beta":self.config.beta}
            self.model = getattr(models, self.config.mixing.lower())(vaes, self.config.n_latents, obj_cfg, self.config.model_cfg)
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
        loss_d = self.model.objective(train_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("train_{}".format(key), loss_d[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss_d[key]):
                    self.log("Mod_{}_TrainLoss".format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def validation_step(self, val_batch, batch_idx):
        """
        Iterates over the test loader
        """
        loss_d = self.model.objective(val_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("val_{}".format(key), loss_d[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss_d[key]):
                    self.log("Mod_{}_ValLoss".format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def validation_epoch_end(self, outputs):
        """
        Save visualizations

        :param outputs: Loss that comes from validation_step
        :type outputs: torch.tensor
        """
        if (self.trainer.current_epoch) % self.config.viz_freq == 0:
            savepath = os.path.join(self.config.mPath, "visuals/epoch_{}/".format(self.trainer.current_epoch))
            os.makedirs(savepath, exist_ok=True)
            self.analyse_data(savedir=savepath)
            self.save_reconstructions(savedir=savepath)
            self.save_traversals(savedir=savepath)

    def save_reconstructions(self, num_samples=24, savedir=None):
        """
        Save reconstructed data

        :param num_samples: number of samples to take for reconstruction
        :type num_samples: int
        :param savedir: where to save the reconstructions
        :type savedir: str
        """
        def save(output, mods, name=None):
            for k in data.keys():
                if mods[k]["data"] is None:
                    mods.pop(k)
            for key in output.mods.keys():
                recon_list = [x.loc for x in output.mods[key].decoder_dist] if isinstance(output.mods[key].decoder_dist, list)\
                    else output.mods[key].decoder_dist.loc
                data_class = self.trainer.datamodule.datasets[int(key.split("_")[-1])-1]
                p = os.path.join(savedir, "recon_{}_to_{}.png".format(name, data_class.mod_type))
                data_class.save_recons(mods, recon_list, p, self.mod_names)

        data = next(iter(self.trainer.datamodule.predict_dataloader(num_samples)))
        data_i = check_input_unpacked(data_to_device(data, self.device))
        save(self.model.forward(data_i), data, "all")
        for m in range(len(data.keys())):
            mods = copy.deepcopy(data)
            for d in mods.keys():
                mods[d]["data"], mods[d]["masks"] = None, None
            mods["mod_{}".format(m + 1)] = data["mod_{}".format(m + 1)]
            mod_name = self.config.mods[m]["mod_type"]
            output = self.model.forward(check_input_unpacked(mods))
            save(output, copy.deepcopy(mods), mod_name)

    def save_traversals(self, num_samples=36, savedir=None):
        """
        Generate and save traversals for each modality

        :param num_samples: number of samples to generate
        :type num_samples: int
        :param savedir: where to save the reconstructions
        :type savedir: str
        """
        if len(self.config.mods) > 1:
            for i, vae in enumerate(self.model.vaes):
                samples = self.model.vaes[vae].generate_samples(num_samples)
                recons = self.model.vaes[vae].decode({"latents": samples, "masks": None})[0]
                data_class = self.trainer.datamodule.datasets[i]
                p = os.path.join(savedir, "traversals_{}.png".format( data_class.mod_type))
                data_class.save_traversals(recons, p)
        else:
            samples = self.model.generate_samples(num_samples)
            recons = self.model.decode({"latents": samples, "masks": None})[0]
            data_class = self.trainer.datamodule.datasets[0]
            p = os.path.join(savedir, "traversals_{}.png".format(data_class.mod_type))
            data_class.save_traversals(recons, p)

    def analyse_data(self, data=None, labels=None, num_samples=250, path_label="", savedir=None):
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
        :param savedir: where to save the reconstructions
        :type savedir: str
        :return: returns K latent samples for each input
        :rtype: list
        """
        if not data:
            data, labels = self.trainer.datamodule.get_num_samples(num_samples)
        data_i = check_input_unpacked(data_to_device(data, self.device))
        output = self.model.forward(data_i)
        output_dic = output.unpack_values()
        pz = self.model.pz(*[x for x in self.model.pz_params])
        zss_sampled = [pz.sample(torch.Size([1, num_samples])).view(-1, pz.batch_shape[-1]),
               *[zs["latents"].view(-1, zs["latents"].size(-1)) for zs in output_dic["latent_samples"]]]
        kl_df = make_kl_df(output_dic["encoder_dist"], pz)
        if path_label == "" and not self.config.eval_only:
                path_label = "_e_{}".format(self.trainer.current_epoch)
        plot_kls_df(kl_df, os.path.join(savedir, 'kl_distance{}.png'.format(path_label)))
        t_sne([x for x in zss_sampled[1:]], os.path.join(savedir, 't_sne{}.png'.format(path_label)), labels)
