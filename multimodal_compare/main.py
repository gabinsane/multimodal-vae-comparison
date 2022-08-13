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
from utils import Logger, save_model, unpack_data, pad_seq_data, transpose_dataloader


def parse_args():
    """
    Loads the .yml config specified in the --cfg argument. Any additional arguments override the values in the config.
    :return: dict; config
    :rtype: dict
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
    with open(args.cfg) as file:
        config = yaml.safe_load(file)
    for name, value in vars(args).items():
        if value is not None and name != "cfg" and name in config.keys():
            config[name] = value
    return config


class Trainer():
    """
    Multimodal VAE trainer common for all architectures. Configures, trains and tests the model.

    :param cfg: parsed .yml config
    :type cfg: dict
    :param device: cuda/cpu
    :type device: object
    """
    def __init__(self, cfg, device):
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
        """
        Initiates the model, get dataloaders, find the selected objective
        """
        self._get_mods_config()
        self._get_model()
        self._setup_savedir()
        self._configure_optimizer()
        self.train_loader, self.test_loader = self.model.load_dataset(self.config["batch_size"], device=self.device)
        self.objective = getattr(objectives, ('multimodal_' if hasattr(self.model, 'vaes') else '')
                                 + ("_".join((self.config["obj"], self.config["mixing"])) if hasattr(self.model, 'vaes')
                                    else self.config["obj"]))

    def _setup_savedir(self):
        """
        Creates the model directory in the results folder and saves the config copy
        """
        self.mPath = os.path.join('results/', self.config["exp_name"])
        os.makedirs(self.mPath, exist_ok=True)
        os.makedirs(os.path.join(self.mPath, "visuals"), exist_ok=True)
        print('Experiment path:', self.mPath)
        with open('{}/config.json'.format(self.mPath), 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def _configure_optimizer(self):
        """
        Sets up the optimizer specified in the config
        """
        assert self.config["optimizer"].lower() in ["adam", "adabelief"], "unsupported optimizer"
        if self.config["optimizer"].lower() == "adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=float(self.config["lr"]), amsgrad=True)
        elif self.config["optimizer"].lower() == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(self.model.parameters(), lr=float(self.config["lr"]), eps=1e-16,
                                       betas=(0.9, 0.999),
                                       weight_decouple=True, rectify=False, print_change_log=False)

    def _get_model(self):
        """
        Sets up the model according to the config file
        """
        model = "VAE" if len(self.mods) == 1 else self.config["mixing"].lower()
        m = getattr(models, model)
        params = [[m["encoder"] for m in self.mods], [m["decoder"] for m in self.mods], [m["path"] for m in self.mods],
                  [m["feature_dim"] for m in self.mods], [m["mod_type"] for m in self.mods]]
        if len(self.mods) == 1:
            params = [x[0] for x in params]
        self.model = m(*params, self.config["n_latents"], self.config["test_split"], self.config["batch_size"]).to(self.device)

        if self.config["pre_trained"]:
            print('Loading model {} from {}'.format(model.modelName, self.config["pre_trained"]))
            self.model.load_state_dict(torch.load(self.config["pre_trained"] + '/model.rar'))
            self.model._pz_params = model._pz_params

    def _get_mods_config(self):
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

    def prepare_data(self, data):
        """
        Unpacks the data sent in the batch for training/testing

        :rtype: object
        :param data: data coming from the DataLoader iterator
        :return: data prepared for training/testing, (int) length of the batch
        """
        if any(["transformer" in x["encoder"].lower() for x in self.mods]):
            data, masks = data
            data = pad_seq_data(list(data), masks) if len(self.mods) > 1 else [data.to(self.device), masks]
            data_len = len(data[0])
        else:
            data = unpack_data(data, device=self.device) \
                if len(self.mods) > 1 else unpack_data(data[0], device=self.device)
            data_len = len(data) if len(self.mods) == 1 else len(data[0])
        return data, data_len

    def prepare_testset(self, num_samples=None):
        """
        Retrieves the TestLoader and prepares it for visualization methods

        :param num_samples: how many samples to prepare
        :type num_samples: int
        Returns:
            :return data: (list)
            :return d_len: (int)
        """
        full_len = len(self.test_loader.dataset)
        if any(["transformer" in x["encoder"].lower() for x in self.mods]):
            data = self.model.seq_collate_fn(self.test_loader.dataset)
            data, d_len = self.prepare_data(data)
            new_set = []
            for x in data:
                if len(x) == full_len:
                    new_set.append(x[:num_samples])
                else:
                    new_set.append([p[:num_samples] for p in x])
            data = new_set
        else:
            if len(self.mods) > 1:
                data = transpose_dataloader(self.test_loader.dataset, device=self.device)
                data = [x[:num_samples] for x in data]
                d_len = len(data[0])
            else:
                data = self.prepare_data(self.test_loader.dataset.tensors)
                data, d_len = data[:num_samples]
        d_len = num_samples if num_samples is not None else d_len
        return data, d_len

    def iterate_epochs(self):
        """
        The main training and testing iterator.
        """
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
        print("Training finished. Elapsed time: {}".format(timedelta(seconds=t1 - t0)))

    def train(self, epoch):
        """
        Iterates over the train loader

        :param epoch: current epoch
        :type epoch: int
        """
        self.model.train()
        loss_m = []
        kld_m = []
        partial_losses = [[] for _ in range(len(self.mods))]
        for it, dataT in enumerate(self.train_loader):
            data, _ = self.prepare_data(dataT)
            self.optimizer.zero_grad()
            loss, kld, partial_l = self.objective(self.model, data, ltype=self.config["loss"])
            loss_m.append(loss)
            kld_m.append(kld)
            for i, l in enumerate(partial_l):
                partial_losses[i].append(l)
            loss.backward()
            self.optimizer.step()
            print("Training iteration {}/{}, loss: {}".format(it, int(
                len(self.train_loader.dataset) / self.config["batch_size"]), float(loss) / self.config["batch_size"]))
        progress_d = {"Epoch": epoch, "Train Loss": self.get_loss_mean(loss_m), "Train KLD": self.get_loss_mean(kld_m)}
        for i, x in enumerate(partial_losses):
            progress_d["Train Mod_{}".format(i)] = self.get_loss_mean(x)
        self.lossmeter.update_train(progress_d)
        print('=====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, self.get_loss_mean(loss_m)))

    def test(self, epoch):
        """
        Iterates over the test loader

        :param epoch: current epoch
        :type epoch: int
        """
        self.model.eval()
        loss_m = []
        kld_m = []
        partial_losses = [[] for _ in range(len(self.mods))]
        with torch.no_grad():
            for ix, dataT in enumerate(self.test_loader):
                data, d_len = self.prepare_data(dataT)
                loss, kld, partial_l = self.objective(self.model, data, ltype=self.config["loss"])
                loss_m.append(loss)
                kld_m.append(kld)
                for i, l in enumerate(partial_l):
                    partial_losses[i].append(l)
                if ix == 0 and epoch % self.config["viz_freq"] == 0:
                    self.model.reconstruct(data, self.mPath, epoch)
                    self.model.generate(self.mPath, epoch)
        if epoch % self.config["viz_freq"] == 0:
            self.visualize_latents(epoch)
        progress_d = {"Epoch": epoch, "Test Loss": self.get_loss_mean(loss_m), "Test KLD": self.get_loss_mean(kld_m)}
        for i, x in enumerate(partial_losses):
            progress_d["Test Mod_{}".format(i)] = self.get_loss_mean(x)
        self.lossmeter.update(progress_d)
        print('====>             Test loss: {:.4f}'.format(self.get_loss_mean(loss_m)))

    def visualize_latents(self, epoch):
        """
        Runs the model analysis, saves T-SNE and KL-Divergences

        :param epoch: current epoch
        :type epoch: int
        """
        testset, testset_len = self.prepare_testset(num_samples=250)
        if self.labels:
            lrange = int(len(self.labels) * (1 - self.config["test_split"]))
            self.model.analyse(testset, self.mPath, epoch, self.labels[lrange:lrange + testset_len])
        else:
            self.model.analyse(testset, self.mPath, epoch, labels=None)

    def get_torch_mean(self, loss):
        """
        Get the mean of the list of torch tensors

        :param loss: list of loss tensors
        :type loss: list
        :return: mean of the losses
        :rtype: torch.float32
        """
        return round(float(torch.mean(torch.tensor(loss).detach().cpu())), 3)


if __name__ == '__main__':
    config = parse_args()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config, dev)
    trainer.iterate_epochs()
