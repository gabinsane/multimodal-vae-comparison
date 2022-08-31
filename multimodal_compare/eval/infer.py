import argparse
import glob
from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import pickle
from models.trainer import MultimodalVAE
from models.config_cls import Config
from models.mmvae_base import TorchMMVAE
from utils import unpack_data, one_hot_encode, output_onehot2text, pad_seq_data, get_root_folder
from models.dataloader import DataModule


class MMVAEExperiment():
    """
    This class provides unified access to all assets of MMVAE eval experiments, i.e., trained models, training stats, config.
    """

    def __init__(self, path):
        """
        Initialize MMVAEExperiment class
        :param path: path to the model checkpoint .ckpt file
        :type path: str
        """
        assert os.path.exists(path), f"{path} does not exist."

        self.base_path = None
        self.base_path = self.get_base_path(path)
        assert os.path.isfile(os.path.join(self.base_path,
                                           'config.yml')), f"Directory {path} does not contain a config."
        self.path = path
        self.config = None
        self.mods = None
        self.data = None
        self.test_loader = None
        self.model = self.load_model_ckpt()

    def get_base_path(self, path_within_exp_folder=None):
        if self.base_path is not None:
            return self.base_path
        else:
            if not os.path.exists(path_within_exp_folder):
                raise ValueError(f"Path is not valid: {path_within_exp_folder}")
            base_path = None
            if os.path.isdir(path_within_exp_folder):
                base_path = path_within_exp_folder
            else:
                base_path = os.path.dirname(path_within_exp_folder)

            # base path should contain a file config.yml and be under the results folder
            while 'results' in base_path:
                if os.path.exists(os.path.join(base_path, 'config.yml')):
                    self.base_path = base_path
                    break
                else:
                    base_path = os.path.dirname(base_path)

            if self.base_path is None:
                raise ValueError(f"{path_within_exp_folder} is not within an exp folder")
            return self.base_path


    def set_data(self, dataloader):
        self.data = dataloader

    def get_data(self):
        if self.data is None:
            raise ValueError(
                'No data is specified. You can load the test data with exp.set_data(exp.get_model_train_test_data()[1])')
        return self.data

    def get_model_test_data(self):
        datamodule = DataModule(self.get_config())
        datamodule.setup()
        self.test_loader = datamodule.val_dataloader()
        return self.test_loader

    def get_model(self):
        if self.model is not None:
            return self.model
        self.model = self.load_model_ckpt()
        return self.model

    def load_model_ckpt(self):
        """
        Loads a model from checkpoint

        :param pth: path to checkpoint directory
        :type pth: str
        :return: returns model object
        :rtype: MMVAE/VAE
        """
        config = self.get_config()
        model = MultimodalVAE(config)
        model_loaded = model.load_from_checkpoint(self.path, cfg=config)
        model_loaded.eval()
        self.model = model_loaded
        return self.model

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
        pth = os.path.join(self.get_base_path(), 'config.yml')
        self.config = Config(pth, eval_only=True)
        return self.config

    def make_evals(self):
        for evaluation in Evaluations.__subclasses__():
            if evaluation.isApplicable(self):
                evaluation.evaluation(self)

    def eval_sample(self):
        """
        Generates random samples from the learned posterior and saves them in the model directory
        :param path: path to the model
        :type path: str
        """
        model = self.get_model()
        assert isinstance(model, TorchMMVAE)
        N = 36
        samples = [s.cpu() for s in model.generate_samples(N)]

        with open(os.path.join(self.path, "visuals/latent_samples.pkl"), 'wb') as handle:
            pickle.dump(np.asarray(samples), handle)
        print("Saved samples for {}".format(self.path))

    def get_device(self):
        return 'cuda'

    def get_traversal_samples(self, latent_dim=None, n_samples_per_dim=1):
        """
        Generates random sample traversals across the whole latent space.
        :param latent_dim: dimensionality of the latent space
        :type latent_dim: int
        :param n_samples_per_dim: how many samples to make per dimension (they will be equally distributed)
        :type n_samples_per_dim: int
        :return: torch tensor samples
        :rtype: torch.tensor
        """
        if latent_dim is None:
            latent_dim = exp.get_model().vaes[-1].n_latents

        all_samples = []
        for idx in range(latent_dim):
            samples = torch.zeros(n_samples_per_dim, latent_dim)
            traversals = torch.linspace(-3, 3, steps=n_samples_per_dim)
            for i in range(n_samples_per_dim):
                samples[i, idx] = traversals[i]
            all_samples.append(samples)
        samples = torch.cat(all_samples)
        return samples

    def image_to_text(self, imgs):
        """
        Reconstructs image from the text input using the provided model
        :param imgs: list of images to reconstruct
        :type imgs: list
        :param model: model object
        :type model: object
        :param path: where to save the outputs
        :type path: str
        :return: returns reconstructed images and texts
        :rtype: tuple(list, list)
        """
        model = self.get_model()
        path = os.path.join(self.get_base_path(), 'image_to_text/')
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

    def text_to_image(self, text):
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
        model = self.get_model()
        path = os.path.join(self.get_base_path(), 'text_to_image/')
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

    def plot_loss(self):
        """
        Plots the Test loss saved in loss.csv in the provided model directory.
        """
        path = self.get_base_path()
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


class Evaluations(ABC):
    """
    Base class to register all evaluation methods with the experiment class.
    """

    @classmethod
    def evaluation(cls, exp: MMVAEExperiment):
        pass

    @classmethod
    def isApplicable(cls, exp):
        return False


class EstimateLogMarginal(Evaluations):
    @classmethod
    def evaluation(self, exp: MMVAEExperiment):
        """
        Estimate of the log-marginal likelihood of test data.
        :param exp: Experiment object that specifies model, data, config, device etc.
        :type exp: MMVAEExperiment
        :return: marginal log-likelihood
        :rtype: float32
        """
        model = exp.get_model()
        test_data = exp.get_data()
        marginal_loglik = 0
        objective = model.objective
        with torch.no_grad():
            for dataTs in test_data:
                # for mod, dataT in dataTs.items():
                #     data, masks = dataT['data'], dataT['masks']
                #     xo = [o[0].clone().detach() for o in data[1][0]]
                #     data = pad_seq_data(data, masks)
                marginal_loglik += objective(model.model, dataTs, ltype="lprob")[0]
        marginal_loglik /= len(test_data.dataset)
        print(f'The marginal loglikelihood is: {marginal_loglik}')
        return marginal_loglik

    @classmethod
    def isApplicable(cls, exp):
        return True


class EvalReconstruct(Evaluations):
    @classmethod
    def evaluation(cls, exp: MMVAEExperiment):
        """
        Makes reconstructions from the testloader for the given model and dumps them into pickle
        :param path: path to the model directory
        :type path: str
        :param testloader: Testloader (torch.utils.data.DataLoader)
        :type testloader: object
        """
        testloader = exp.get_data()
        model = exp.get_model()
        device = torch.device("cuda")
        recons = []
        for i, data in enumerate(testloader):
            #d = unpack_data(data['mod_1'], device=device)
            #recon = model.reconstruct_data(d)
            recon = model.model.forward(data)
            recons.append([np.asarray(data[mod]['data'].detach().cpu()) for mod in data.keys()])
            recons.append([recon[mod].decoder_dists for mod in data.keys()])
            # recons.append(np.asarray(recon[0].detach().cpu())[0])
            if i == 2:
                break
        config = exp.get_config()
        path = exp.get_base_path()
        with open(os.path.join(path, "visuals/reconstructions.pkl"), 'wb') as handle:
            pickle.dump(recons, handle)
        print("Saved reconstructions for {}".format(path))

    @classmethod
    def isApplicable(cls, exp):
        return True


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


def compare_loss(exps, label_tag):
    """
    Compares losses for several models in one plot.
    :param exps: list of experiment objects
    :type exps: list[MMVAEExperiment]
    :param label_tag: list of strings to label the models in the legend
    :type label_tag: list
    """
    for ll in ["Test Loss", "Test Mod_0", "Test Mod_1"]:
        for p in exps:
            pth = os.path.join(p.get_base_path(), "loss.csv") if "loss.csv" not in p else p
            cfg = p.get_config()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=str, help="name of the experiment directory")
    args = parser.parse_args()
    if args.exp:
        model = args.exp
    else:
        model = 'test'

    # get all checkpoints in the folder and select the latest one
    p1 = os.path.join(get_root_folder(), f"results/{model}/lightning_logs/version_*/checkpoints/*.ckpt")
    p2 = os.path.join(get_root_folder(), f"results/{model}/*.ckpt")

    patterns = (p1, p2)
    files_grabbed = []
    for pattern in patterns:
        files_grabbed.extend(glob.glob(pattern))
    if len(files_grabbed) > 0:
        p = max(files_grabbed, key=os.path.getctime)
    else:
        raise FileNotFoundError('No ckpt file available')

    exp = MMVAEExperiment(path=p)
    data = exp.get_model_test_data()
    exp.set_data(data)
    exp.make_evals()
