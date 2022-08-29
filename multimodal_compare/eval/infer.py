import csv
import glob
import json
from abc import ABC, abstractmethod

import yaml
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

import models
import pickle
from models import objectives
from models.mmvae_base import TorchMMVAE
from utils import unpack_data, one_hot_encode, output_onehot2text, pad_seq_data


# We have models, data, objectives


class MMVAEExperiment():
    """
    This class provides unified access to all assets of MMVAE experiments, i.e., trained models, training stats, config.
    """

    def __init__(self, path):
        """
        Initialize MMVAEExperiment class

        :param path: path to the experiment directory
        :type path: str
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."
        assert os.path.isfile(os.path.join(path, 'model.rar')), f"Directory {path} does not contain a model."
        assert os.path.isfile(os.path.join(path, 'config.yml')), f"Directory {path} does not contain a config."
        self.path = path

        self.config = None
        self.mods = None
        self.model = None
        self.data = None

    def set_data(self, dataloader):
        self.data = dataloader

    def get_data(self):
        if self.data is None:
            raise ValueError(
                'No data is specified. You can load the test data with exp.set_data(exp.get_model_train_test_data()[1])')
        return self.data

    def get_model_train_test_data(self):
        model = self.get_model()
        train_loader, test_loader = model.load_dataset(32,
                                                       device=self.get_device())  # TODO: This seems to be not general enough
        return train_loader, test_loader

    def get_model(self):
        """
            Loads the model from the experiment directory

            :return: model object
            :rtype: object
            """
        if self.model is not None:
            return self.model
        device = torch.device("cuda")
        config = self.get_config()
        mods = self.get_modalities()
        model_ = "VAE" if len(mods) == 1 else config["mixing"].lower()
        modelC = getattr(models, model_)
        params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods],
                  [m["feature_dim"] for m in mods], ["image", "text"]]
        if len(mods) == 1:
            params = [x[0] for x in params]

        vaes = []
        from models import VAE
        for i, m in enumerate(self.mods):
            vaes.append(VAE(m["encoder"], m["decoder"], m["feature_dim"], self.config['n_latents']))
        model_ = modelC(vaes, config).to(device)


        # model_ = modelC(*params, config["n_latents"], 0.1, config["batch_size"]).to(device)
        print('Loading model {} from {}'.format(model_.modelName, self.path))
        model_.load_state_dict(torch.load(os.path.join(self.path, 'model.rar')))
        model_._pz_params = model_._pz_params
        model_.eval()
        self.model = model_
        return model_

    def get_config(self):
        """
        Parses the YAML config provided in the file path

        :param pth: Path to config
        :type pth: str
        :return: returns the config dict
        :rtype: dict
        """
        if self.config is not None:
            return self.config
        if os.path.isdir(self.path):
            conf_pth = os.path.join(self.path, 'config.json')
        with open(conf_pth, 'r') as stream:
            config = yaml.safe_load(stream)
        self.config = config
        return config

    def get_modalities(self):
        """
        Loads the modalities of a model based on the config.
        :return: returns list of modality-specific dicts
        :rtype: list[]
        """
        if self.mods is not None:
            return self.mods
        config = self.get_config()
        modalities = []
        for x in range(20):
            if "modality_{}".format(x) in list(config.keys()):
                modalities.append(config["modality_{}".format(x)])
        self.mods = modalities
        return modalities

    def make_evals(self):
        for evaluation in Evaluations.__subclasses__():
            evaluation.evaluation()

    def get_device(self):
        return 'cuda'


class Evaluations(ABC):
    """
    Base class to register all evaluation methods with the experiment class.
    """

    @abstractmethod
    def evaluation(self, exp: MMVAEExperiment):
        pass


class EstimateLogMarginal(Evaluations):
    @classmethod
    def evaluation(self, exp:MMVAEExperiment):
        """
        Estimate of the log-marginal likelihood of test data.

        :param exp: Experiment object that specifies model, data, config, device etc.
        :type exp: MMVAEExperiment
        :return: marginal log-likelihood
        :rtype: float32
        """
        model = exp.get_model()
        test_data = exp.get_data()
        assert isinstance(model, TorchMMVAE)

        # train_loader, test_loader = model.load_dataset(32, device=device)
        model.eval()
        marginal_loglik = 0
        objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '')
                            + ("_".join(("elbo", model.modelName)) if hasattr(model, 'vaes') else ["elbo"]))
        with torch.no_grad():
            for ix, dataT in enumerate(test_data):
                data, masks = dataT
                data = pad_seq_data(data, masks)
                marginal_loglik += objective(model, data, ltype="lprob")[0]

        marginal_loglik /= len(test_data.dataset)
        return marginal_loglik


# def parse_args(pth):
#     """
#     Parses the YAML config provided in the file path
#
#     :param pth: Path to config
#     :type pth: str
#     :return: returns the config dict and modality-specific dict
#     :rtype: tuple(dict, dict)
#     """
#     if os.path.isdir(pth):
#         pth = os.path.join(pth, 'config.json')
#     with open(pth, 'r') as stream:
#         config = yaml.safe_load(stream)
#     modalities = []
#     for x in range(20):
#         if "modality_{}".format(x) in list(config.keys()):
#             modalities.append(config["modality_{}".format(x)])
#     return config, modalities


plot_colors = ["blue", "green", "red", "cyan", "magenta", "orange", "navy", "maroon", "brown"]


def estimate_log_marginal(model, device="cuda"):
    """
    Estimate of the log-marginal likelihood of test data.


    :param model: VAE model
    :type model: object
    :param device: device (cuda/cpu)
    :type device: str
    :return: marginal log-likelihood
    :rtype: float32
    """
    train_loader, test_loader = model.load_dataset(32, device=device)
    model.eval()
    marginal_loglik = 0
    objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '')
                        + ("_".join(("elbo", model.modelName)) if hasattr(model, 'vaes') else ["elbo"]))
    with torch.no_grad():
        for ix, dataT in enumerate(test_loader):
            data, masks = dataT
            data = pad_seq_data(data, masks)
            marginal_loglik += objective(model, data, ltype="lprob")[0]

    marginal_loglik /= len(test_loader.dataset)
    return marginal_loglik


def eval_reconstruct(path, testloader=None):
    """
    Makes reconstructions from the testloader for the given model and dumps them into pickle

    :param path: path to the model directory
    :type path: str
    :param testloader: Testloader (torch.utils.data.DataLoader)
    :type testloader: object
    """
    if testloader is None:
        # make other loader
        pass
    model = load_model(path)
    device = torch.device("cuda")
    recons = []
    for i, data in enumerate(testloader):
        d = unpack_data(data[0], device=device)
        recon = model.reconstruct_data(d)
        recons.append(np.asarray(d[0].detach().cpu()))
        recons.append(np.asarray(recon[0].detach().cpu())[0])
        if i == 10:
            break
    with open(os.path.join(path, "visuals/reconstructions.pkl"), 'wb') as handle:
        pickle.dump(recons, handle)
    print("Saved reconstructions for {}".format(path))


def eval_sample(path):
    """
    Generates random samples from the learned posterior and saves them in the model directory

    :param path: path to the model
    :type path: str
    """
    model = load_model(path)
    assert isinstance(model, TorchMMVAE)
    N = 36
    samples = [s.cpu() for s in model.generate_samples(N)]

    with open(os.path.join(path, "visuals/latent_samples.pkl"), 'wb') as handle:
        pickle.dump(np.asarray(samples), handle)
    print("Saved samples for {}".format(path))


def _plot_setup(xname, yname, pth, figname):
    """
    General plot set up functions

    :param xname: Name of x axis
    :type xname: str
    :param yname: Name of y axis
    :type yname: str
    :param pth: path to save the plot to
    :type pth: str
    :param figname: name of the figure
    :type figname: str
    """
    plt.xlabel(xname)
    plt.ylabel(yname)
    p = pth if os.path.isdir(pth) else os.path.dirname(pth)
    plt.savefig(os.path.join(p, "visuals/{}.png".format(figname)))
    plt.clf()


def plot_loss(path):
    """
    Plots the Test loss saved in loss.csv in the provided model directory.

    :param path: path to model directory
    :type path: str
    """
    pth = os.path.join(path, "loss.csv") if not "loss.csv" in path else path
    loss = pd.read_csv(pth, delimiter=",")
    epochs = loss["Epoch"]
    losses = loss["Test Loss"]
    kld = loss["Test KLD"]
    plt.plot(epochs, losses, color='green', linestyle='solid', label="Loss")
    _plot_setup("Epochs", "Loss", path, "loss")
    plt.plot(epochs, kld, color='blue', linestyle='solid', label="KLD")
    _plot_setup("Epochs", "KLD", path, "KLD")
    print("Saved loss plot")


def compare_loss(paths, label_tag):
    """
    Compares losses for several models in one plot.

    :param paths: list of paths to model directories
    :type paths: list
    :param label_tag: list of strings to label the models in the legend
    :type label_tag: list
    """
    for ll in ["Test Loss", "Test Mod_0", "Test Mod_1"]:
        for p in paths:
            pth = os.path.join(p, "loss.csv") if "loss.csv" not in p else p
            c = os.path.join(p, "config.json") if "loss.csv" not in p else p.replace("loss.csv", "config.json")
            with open(c) as json_file:
                cfg = json.load(json_file)
            loss = pd.read_csv(pth, delimiter=",")
            epochs = loss["Epoch"]
            losses = loss[ll]
            plt.plot(epochs, losses, linestyle='solid',
                     label=", ".join(["{}: {}".format(x, str(cfg[x])) for x in label_tag]))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(paths[0])),
                                 "losstype_compared_{}.png".format(ll.lower().replace(" ", "_"))))
        plt.clf()


def _get_all_csvs(pth):
    """
    Extracts paths to all csv files within a directory and its subdirectories

    :param pth: path to directory
    :type pth: str
    :return: list of loss paths
    :rtype: list
    """
    pth = pth + "/" if pth[-1] != "/" else pth
    return glob.glob(pth + "**/loss.csv", recursive=True)


def compare_models_numbers(pth):
    """
    Compares losses for multiple models and prints it into a text file

    :param pth: list of model directories to compare
    :type pth: list
    """
    f = open(os.path.join(pth[0], "comparison.csv"), 'w')
    writer = csv.writer(f)
    all_csvs = _get_all_csvs(pth)
    header = None
    for c in all_csvs:
        model_csv = pd.read_csv(c, delimiter=",")
        if not header:
            writer.writerow(["Model", "Epochs"] + list(model_csv.keys())[1:])
            header = True
        row = [c.split(pth)[-1].split("/loss.csv")[0]] + [int(model_csv.values[-1][0])] + [round(x, 4) for x in list(
            model_csv.values[-1][1:])]
        writer.writerow(row)
    f.close()
    print("Saved comparison at {}".format(os.path.join(pth[0], "comparison.csv")))


# def load_model(path):
#     """
#     Loads the model from directory path
#
#     :param path: path to model directory
#     :type path: str
#     :return: model object
#     :rtype: object
#     """
#     device = torch.device("cuda")
#     config, mods = parse_args(path)
#     model_ = "VAE" if len(mods) == 1 else config["mixing"].lower()
#     modelC = getattr(models, model_)
#     params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods],
#               [m["feature_dim"] for m in mods], ["image", "text"]]
#     if len(mods) == 1:
#         params = [x[0] for x in params]
#     model_ = modelC(*params, config["n_latents"], 0.1, config["batch_size"]).to(device)
#     print('Loading model {} from {}'.format(model_.modelName, path))
#     model_.load_state_dict(torch.load(os.path.join(path, 'model.rar')))
#     model_._pz_params = model_._pz_params
#     model_.eval()
#     return model_


def get_traversal_samples(latent_dim, n_samples_per_dim):
    """
    Generates random sample traversals across the whole latent space.

    :param latent_dim: dimensionality of the latent space
    :type latent_dim: int
    :param n_samples_per_dim: how many samples to make per dimension (they will be equally distributed)
    :type n_samples_per_dim: int
    :return: torch tensor samples
    :rtype: torch.tensor
    """
    all_samples = []
    for idx in range(latent_dim):
        samples = torch.zeros(n_samples_per_dim, latent_dim)
        traversals = torch.linspace(-3, 3, steps=n_samples_per_dim)
        for i in range(n_samples_per_dim):
            samples[i, idx] = traversals[i]
        all_samples.append(samples)
    samples = torch.cat(all_samples)
    return samples


def text_to_image(text, model, path):
    """
    Reconstructs text from the image input using the provided model

    :param text: list of strings to reconstruct
    :type text: list
    :param model: model object
    :type model: object
    :param path: where to save the outputs
    :type path: str
    :return: returns reconstructed images and also texts
    :rtype: tuple(list, list)
    """
    img_outputs, txtoutputs = [], []
    for i, w in enumerate(text):
        txt_inp = one_hot_encode(len(w), w.lower())
        model.eval()
        recons = model.forward([None, [txt_inp.unsqueeze(0), None]])[1]
        if model.modelName == 'moe':
            recons = recons[0]
        image, recon_text = _process_recons(recons)
        txtoutputs.append(recon_text[0][0])
        img_outputs.append(image * 255)
        image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, "cross_sample_{}_{}.png".format(recon_text[0][0][:len(w)], i)), image)
    return img_outputs, txtoutputs


def _process_recons(recons):
    """
    Processes the reconstructions from the model so that they can be visualized

    :param recons: list of the model outputs
    :type recons: list
    :return: image and text
    :rtype: tuple(ndarray, string)
    """
    recons_image = recons[0] if isinstance(recons, list) else recons
    recons_image = recons_image[0] if isinstance(recons_image, list) else recons_image
    recons_text = recons[1][1] if isinstance(recons[1], list) else recons[1]
    image = np.asarray(recons_image.loc[0].cpu().detach())
    recon_text = recons_text.loc[0]
    recon_text = output_onehot2text(recon=recon_text.unsqueeze(0))
    return image, recon_text


def image_to_text(imgs, model, path):
    """
    Reconstructs image from the text input using the provided model

    :param text: list of strings to reconstruct
    :type text: list
    :param model: model object
    :type model: object
    :param path: where to save the outputs
    :type path: str
    :return: returns reconstructed images and texts
    :rtype: tuple(list, list)
    """
    txt_outputs, img_outputs = [], []
    model.eval()
    for i, w in enumerate(imgs):
        recons = model.forward([w.unsqueeze(0), None])[1]
        image, recon_text = _process_recons(recons)
        txt_outputs.append(recon_text[0][0])
        img_outputs.append(image * 255)
        image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, "cross_sample_{}_{}.png".format(recon_text[0][0][:len(w)], i)), image)
    return img_outputs, txt_outputs


def _listdirs(rootdir):
    """
    Lists all subdirectories within a directory

    :param rootdir: root directory path
    :type rootdir: str
    :return: list of subdirectories
    :rtype: list
    """
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


if __name__ == "__main__":
    p = "../results/test"
    exp = MMVAEExperiment(path=p)

    # model = exp.get_model()
    exp.set_data(exp.get_model_train_test_data()[1])
    exp.make_evals()

    # model = load_model(p)
    t = ["pieslice", "circle", "spiral", "line", "square", "semicircle", "pieslice", "circle", "spiral", "line",
         "square", "semicircle", "pieslice", "circle", "spiral", "line", "square", "semicircle"]
    # text_to_image(t, model, p)

    print('still here')
