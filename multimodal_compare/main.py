import argparse
import yaml
import pickle
import numpy as np
import torch
from torch import optim
import os
from timeit import default_timer as timer
from datetime import timedelta
from eval.infer import plot_loss, eval_reconstruct, eval_sample
import models
from models import objectives
from utils import Logger, save_model, unpack_data, pad_seq_data


def parse_args():
    """
    Loads the .yml config specified in the --cfg argument. Any additional arguments override the values in the config.
    :return: dict; config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
    parser.add_argument('--viz_freq', type=int, default=None,
                        help='frequency of visualization savings (number of iterations)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Size of the training batch')
    parser.add_argument('--obj', type=str, metavar='O', default=None,
                        help='objective to use (moe_elbo/poe_elbo_semi)')
    parser.add_argument('--loss', type=str, metavar='O', default=None,
                        help='loss to use (lprob/bce)')
    parser.add_argument('--n_latents', type=int, default=None,
                        help='latent vector dimensionality')
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='path to pre-trained model (train from scratch if empty)')
    parser.add_argument('--seed', type=int, metavar='S', default=None,
                        help='seed number')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='name of folder')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='optimizer')
    args = parser.parse_args()
    with open(args.cfg) as file: config = yaml.safe_load(file)
    for name, value in vars(args).items():
        if value is not None and name != "cfg" and name in config.keys():
            config[name] = value
    return config

class Trainer():
    def __init__(self, cfg, device):
        """
        Multimodal VAE trainer common for all architectures. Configures, trains and tests the model.
        :param cfg: dict, parsed .yml config
        :param device: torch.device object (cuda/cpu)
        """
        self.config = cfg
        self.mods = None
        self.device = device
        self.labels = None
        self.optimizer = None
        self.lossmeter = None
        self.train_loader, self.test_loader = None, None
        self.objective = None
        self.mPath = None
        self.model = None
        self.setup()

    def setup(self):
        self.get_mods_config()
        self.get_model()
        self.setup_savedir()
        self.configure_optimizer()
        self.train_loader, self.test_loader = self.model.load_dataset(self.config["batch_size"], device=self.device)
        self.objective = getattr(objectives, ('multimodal_' if hasattr(self.model, 'vaes') else '')
                            + ("_".join((self.config["obj"], self.config["mixing"])) if hasattr(self.model, 'vaes')
                            else self.config["obj"]))

    def setup_savedir(self):
        self.mPath = os.path.join('results/', self.config["exp_name"])
        os.makedirs(self.mPath, exist_ok=True)
        os.makedirs(os.path.join(self.mPath, "visuals"), exist_ok=True)
        print('Experiment path:', self.mPath)
        with open('{}/config.json'.format(self.mPath), 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def configure_optimizer(self):
        assert self.config["optimizer"].lower() in ["adam", "adabelief"], "unsupported optimizer"
        if self.config["optimizer"].lower() == "adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=float(self.config["lr"]), amsgrad=True)
        elif self.config["optimizer"].lower() == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(self.model.parameters(), lr=float(self.config["lr"]), eps=1e-16, betas=(0.9,0.999),
                                       weight_decouple=True, rectify=False, print_change_log=False)

    def get_model(self):
        model = "VAE" if len(self.mods) == 1 else self.config["mixing"].lower()
        m = getattr(models, model)
        params = [[m["encoder"] for m in self.mods], [m["decoder"] for m in self.mods], [m["path"] for m in self.mods],
                  [m["feature_dim"] for m in self.mods], [m["mod_type"] for m in self.mods]]
        if len(self.mods) == 1:
            params = [x[0] for x in params]
        self.model = m(*params, self.config["n_latents"], self.config["batch_size"]).to(self.device)

        if self.config["pre_trained"]:
            print('Loading model {} from {}'.format(model.modelName, self.config["pre_trained"]))
            self.model.load_state_dict(torch.load(self.config["pre_trained"] + '/model.rar'))
            self.model._pz_params = model._pz_params

    def get_mods_config(self):
        """
        Retrieves the modality-specific configs from the .yml config
        """
        self.mods = []
        for x in range(20):
            if "modality_{}".format(x) in list(self.config.keys()):
                self.mods.append(self.config["modality_{}".format(x)])
        if self.config["labels"]:
            with open(self.config["labels"], 'rb') as handle:
                self.labels = pickle.load(handle)

    def iterate_epochs(self):
        self.lossmeter = Logger(trainer.mPath, self.mods)
        t0 = timer()
        for epoch in range(1, int(self.config["epochs"]) + 1):
            self.train(epoch)
            self.test(epoch)
            save_model(self.model, self.mPath + '/model.rar')
        plot_loss(self.mPath)
        eval_sample(self.mPath)
        eval_reconstruct(self.mPath)
        t1 = timer()
        print("Training finished. Elapsed time: {}".format(timedelta(seconds=t1-t0)))


    def train(self, epoch):
        """
        Iterates over the train loader
        :param epoch: (int) Current epoch
        """
        self.model.train()
        loss_m = []
        kld_m = []
        partial_losses =  [[] for _ in range(len(self.mods))]
        for it, dataT in enumerate(self.train_loader):
            if len(self.mods) > 1:
                if not isinstance(dataT, tuple):
                    data = unpack_data(dataT, device=self.device)
                else:
                    data, masks = dataT
                    data = pad_seq_data(data, masks)
            else:
                if "transformer" in self.config["modality_1"]["encoder"].lower():
                    data, masks = dataT
                    data = [data.to(self.device), masks]
                else:
                    data = unpack_data(dataT[0], device=self.device)
            self.optimizer.zero_grad()
            loss, kld, partial_l = self.objective(self.model, data, ltype=self.config["loss"])
            loss_m.append(loss)
            kld_m.append(kld)
            for i,l in enumerate(partial_l):
                partial_losses[i].append(l)
            loss.backward()
            self.optimizer.step()
            print("Training iteration {}/{}, loss: {}".format(it, int(len(self.train_loader.dataset)/self.config["batch_size"]), float(loss)/self.config["batch_size"]))
        progress_d = {"Epoch": epoch, "Train Loss": self.get_loss_mean(loss_m), "Train KLD": self.get_loss_mean(kld_m)}
        for i, x in enumerate(partial_losses):
            progress_d["Train Mod_{}".format(i)] = self.get_loss_mean(x)
        self.lossmeter.update_train(progress_d)
        print('=====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, self.get_loss_mean(loss_m)))

    def test(self, epoch):
        """
        Iterates over the test loader
        :param epoch: (int) Current epoch
        """
        self.model.eval()
        loss_m = []
        kld_m = []
        partial_losses =  [[] for _ in range(len(self.mods))]
        with torch.no_grad():
            for ix, dataT in enumerate(self.test_loader):
                if len(self.mods) > 1:
                    if not isinstance(dataT, tuple):
                        data = unpack_data(dataT, device=self.device)
                        d_len = len(data[0])
                    else:
                        data, masks = dataT
                        data = pad_seq_data(data, masks)
                        d_len = len(data[0])
                else:
                    if "transformer" in self.config["modality_1"]["encoder"].lower():
                        data, masks = dataT
                        data = [data.to(self.device), masks]
                        d_len = len(data[0])
                    else:
                        data = unpack_data(dataT[0], device=self.device)
                        d_len = len(data)
                loss, kld, partial_l = self.objective(self.model, data, ltype=self.config["loss"])
                loss_m.append(loss)
                kld_m.append(kld)
                for i, l in enumerate(partial_l):
                    partial_losses[i].append(l)
                if ix == 0 and epoch % self.config["viz_freq"] == 0:
                    self.model.reconstruct(data, self.mPath, epoch)
                    self.model.generate(self.mPath, epoch)
                    if self.labels:
                         self.model.analyse(data, self.mPath, epoch,
                                            self.labels[int(len(self.labels)*0.9):int(len(self.labels)*0.9)+d_len])
                    else:
                         self.model.analyse(data, self.mPath, epoch, labels=None)
        progress_d = {"Epoch": epoch, "Test Loss": self.get_loss_mean(loss_m), "Test KLD": self.get_loss_mean(kld_m)}
        for i, x in enumerate(partial_losses):
            progress_d["Test Mod_{}".format(i)] = self.get_loss_mean(x)
        self.lossmeter.update(progress_d)
        print('====>             Test loss: {:.4f}'.format(self.get_loss_mean(loss_m)))

    def get_loss_mean(self, loss):
        """
        Get the mean from the list of loss tensors
        :param loss: list of loss tensors
        :return: float; mean of the losses
        """
        return round(float(torch.mean(torch.tensor(loss).detach().cpu())),3)


if __name__ == '__main__':
    config = parse_args()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    #torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config, dev)
    trainer.iterate_epochs()

