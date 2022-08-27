# Base VAE class definition
import math
import os
import pickle

import cv2
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_proc.process_audio import numpy_to_wav
from models import encoders, decoders
from utils import get_mean, kl_divergence, load_images, lengths_to_mask
from utils import one_hot_encode, output_onehot2text
from visualization import t_sne, tensors_to_df, plot_kls_df


class VaeDataset(object):
    """

    """
    def __init__(self, pth, data_dim, network_type, network_name, mod_type):
        """
        Class for dataset loading and adjustments for training

        :param pth: path to the modality data
        :type pth: str
        :param data_dim: dimensions of the modality, e.g. [64,64,3]
        :type data_dim: list
        :param network_type: net_type parameter of the encoder/decoder
        :type network_type: str
        :param network_name: name of the encoder/decoder class
        :type network_name: str
        :param mod_type: how to treat the modality (image/text)
        :type mod_type: str
        """
        self.pth = pth
        self.data_dim = data_dim
        self.network_type = network_type
        self.network_name = network_name
        self.mod_type = mod_type

    def get_path_type(self, path):
        """
        See if the provided data path is supported.

        :param path: Path to the dataset
        :type path: str
        :return: recognised type of the data
        :rtype: str
        """
        assert os.path.exists(path), "Path does not exist: {}".format(path)
        if os.path.isdir(path):
            return "dir"
        if path[-4:] == ".pth":
            return "torch"
        if path[-4:] == ".pkl":
            return "pickle"
        raise Exception("Unrecognized dataset format. Supported types are: .pkl, .pth or directory with images")

    def load_data(self):
        """
        Loads the data from path

        :return: data prepared for training
        :rtype: torch.tensor
        """
        dtype = self.get_path_type(self.pth)
        if dtype == "dir":
            d = load_images(self.pth, self.data_dim)
        elif dtype == "torch":
            d = torch.load(self.pth)
        elif dtype == "pickle":
            with open(self.pth, 'rb') as handle:
                 d = pickle.load(handle)
        d = self.prepare_for_encoder(d)
        return d

    def check_img_normalize(self, data):
        """
        Normalizes image data between 0 and 1 (if needed)

        :param data: input images for normalisation
        :type data: Union[torch.tensor, list]
        :return: normalised data
        :rtype: list, torch.tensor
        """
        if isinstance(data, list):
            if torch.max(torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)) > 1:
                data = [x/256 for x in data]
        else:
            data = data/256 if torch.max(data) > 1 else data
        return data

    def prepare_for_encoder(self, data):
        """
        Prepares the data for training.

        :param data: the loaded data
        :type data: Union[list, torch.tensor, ndarray]
        :return: data reshaped for training,
        :rtype: torch.tensor
        """
        if self.network_type.lower() in ["transformer", "cnn", "3dcnn", "fnn"]:
            data = [torch.from_numpy(np.asarray(x).astype(np.float)) for x in data]
            if self.network_type in ["cnn", "fnn"]:
                data = torch.stack(data).transpose(1,3)
            if "transformer" in self.network_type.lower():
                if len(data[0].shape) < 3:
                    data = [torch.unsqueeze(i, dim=1) for i in data]
        elif "text" in self.mod_type:
            if len(data[0]) > 1 and not isinstance(data[0], str):
                data = [" ".join(x) for x in data] if not "cub_" in self.pth else data
            data = [one_hot_encode(len(f), f) for f in data]
            data = [torch.from_numpy(np.asarray(x)) for x in data]
            if "transformer" not in self.network_type.lower():
                data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        if self.network_type.lower() == "audioconv":
            self.prepare_audio(data)
        if "image" in self.mod_type:
            data = self.check_img_normalize(data)
        return data

    def prepare_audio(self, data):
        """
        @TODO Support for audio data

        :param data: input audio sequences
        :type data: list
        :return: padded sequences for training
        :rtype: torch.tensor
        """

        d = [torch.from_numpy(np.asarray(x).astype(np.int16)) for x in data]
        return torch.nn.utils.rnn.pad_packed_sequence(d, batch_first=True, padding_value=0.0)

    def get_train_test_splits(self, data, test_fraction):
        """
        Returns the data split into train and validation set according to test_fraction

        :param data: input data
        :type data: Union[torch.tensor, ndarray]
        :param test_fraction: fraction of the data that will be used for test set
        :type test_fraction: float32
        :return: tuple(train_split, test_split)
        """
        train_split = data[:int(len(data)*(1 - test_fraction))]
        test_split = data[int(len(data)*(1-test_fraction)):]
        if self.network_name.lower() not in ["transformer","txttransformer", "3dcnn"]:
            train_split = torch.utils.data.TensorDataset(torch.stack(train_split).clone().detach())
            test_split = torch.utils.data.TensorDataset(torch.stack(test_split).clone().detach())
        return train_split, test_split


class VAE(nn.Module):
    def __init__(self, enc, dec, data_path, feature_dim, mod_type, n_latents, test_split, batch_size,
                 prior_dist=dist.Normal, likelihood_dist=dist.Normal, post_dist=dist.Normal):
        """
        The general unimodal VAE module, can be used separately or in a multimodal VAE

        :param enc: encoder name
        :type enc: str
        :param dec: decoder name
        :type dec: str
        :param data_path: path to training data
        :type data_path: str
        :param feature_dim: data dimensionality as stated in config
        :type feature_dim: list
        :param mod_type: modality type (image/text)
        :type mod_type: str
        :param n_latents: latent space dimensionality
        :type n_latents: int
        :param test_split: fraction of data to be used for testing
        :type test_split: float
        :param batch_size: batch size
        :type batch_size: int
        :param prior_dist: prior distribution
        :type prior_dist: torch.dist
        :param likelihood_dist: likelihood distribution
        :type likelihood_dist: torch.dist
        :param post_dist: posterior distribution
        :type post_dist: torch.dist
        """
        super(VAE, self).__init__()
        self.device = None
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.batch_size = batch_size
        self._qz_x_params = None
        self.llik_scaling = 1.0
        self.test_split = test_split
        self.pth = data_path
        self.mod_type = mod_type
        self.data_dim = feature_dim
        self.n_latents = n_latents
        self.enc_name, self.dec_name = enc, dec
        self.enc, self.dec = self.get_nework_classes(enc, dec)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False)  # logvar
        ])
        self.modelName = 'vae_{}'.format(os.path.basename(data_path))

    def get_nework_classes(self, enc, dec):
       """
        Instantiates the encoder and decoder networks

       :param enc: encoder name
       :type enc: str
       :param dec: decoder name
       :type dec: str
       :return: returns encoder and decoder class
       :rtype: tuple(object, object)
       """
       assert hasattr(encoders, "Enc_{}".format(enc)), "Did not find encoder {}".format(enc)
       enc_obj = getattr(encoders, "Enc_{}".format(enc))(self.n_latents, self.data_dim)
       assert hasattr(decoders, "Dec_{}".format(dec)), "Did not find decoder {}".format(enc)
       dec_obj = getattr(decoders, "Dec_{}".format(dec))(self.n_latents, self.data_dim)
       return enc_obj, dec_obj

    @property
    def pz_params(self):
        """
        :return: returns likelihood parameters
        :rtype: list(torch.tensor, torch.tensor)
        """
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @property
    def qz_x_params(self):
        """
        :return: returns posterior distribution parameters
        :rtype: list(torch.tensor, torch.tensor)
        """
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    def load_dataset(self, batch_size,device="cuda"):
        """
        Loads data and creates DataLoaders

        :param batch_size: batch size
        :type batch_size: int
        :param device: device to put data on
        :type device: str
        :return: train and test split
        :rtype: tuple(torch.DataLoader, torch.DataLoader)
        """
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        dataset_kwargs = {"data_dim":self.data_dim, "network_name":self.enc_name,
                          "network_type":self.enc.net_type, "mod_type":self.mod_type}
        dataset = VaeDataset(self.pth, **dataset_kwargs)
        d = dataset.load_data()
        if "transformer" in self.enc.net_type.lower():
            kwargs["collate_fn"] = self.seq_collate_fn
        train_split, test_split = dataset.get_train_test_splits(d, self.test_split)
        train = DataLoader(train_split, batch_size=batch_size, shuffle=False, **kwargs)
        test = DataLoader(test_split, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

    def forward(self, x, K=1):
        """
        Forward pass

        :param x: input modality
        :type x: torch.tensor
        :param K: sample K samples from the posterior
        :type K: int
        :return: the posterior distribution, the reconstruction and latent samples
        :rtype:tuple(torch.dist, torch.dist, torch.tensor)
        """
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        if "transformer" in self.dec.net_type.lower():
            px_z = self.px_z(*self.dec([zs, x[1]] if x is not None else [zs, None]))
        else: px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def seq_collate_fn(self, batch):
        """
        Collate function for sequential data

        :param batch: list with sequential data
        :return: batch, masks
        """
        new_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in batch]))).to(self.device)
        return new_batch, masks

    def generate(self, runPath, epoch):
        """
        Generate traversals, currently only works for images

        :param runPath: path to save data to
        :type runPath: str
        :param epoch: current epoch to name data
        :type epoch: int
        """
        N, K = 36, 1
        l_s = int(math.sqrt(N))
        samples = self.generate_samples(N, K).cpu().squeeze()
        r_l = []
        if "image" in self.pth:
            for r, recons_list in enumerate(samples):
                    recon = recons_list.cpu().reshape(*self.data_dim).unsqueeze(0)
                    r_l = np.asarray(recon) if r_l == [] else np.concatenate((r_l, np.asarray(recon)))
            rows = []
            for s in range(l_s):
                rows.append(np.hstack(r_l[(s * l_s):(s * l_s) + l_s]))
            r_l = np.vstack(rows)
            cv2.imwrite('{}/visuals/traversals_{:03d}.png'.format(runPath, epoch), r_l*255)

    def reconstruct(self, data, runPath, epoch, N=32):
        """
        Reconstruct data

        :param data: input modality
        :type data: torch.tensor
        :param runPath: path to save data to
        :type runPath: str
        :param epoch: current epoch to name data
        :type epoch: int
        :param N: How many samples to reconstruct
        :type N: int
        """
        recons_mat = self.reconstruct_data(data[:N]).squeeze().cpu()
        if self.mod_type == "image":
            o_l, r_l = [], []
            data = data[0] if "transformer" in self.enc.net_type.lower() else data
            for r, recons_list in enumerate(recons_mat[:N]):
                    _data = data.cpu()[r][:N].reshape(-1, self.data_dim[-3], self.data_dim[-2], self.data_dim[-1]).squeeze()
                    _data = np.hstack(_data) if len(_data.shape) > 3 else _data
                    recon = np.hstack(recons_list.cpu()) if len(recons_list.cpu().shape) > 3 else recons_list.cpu()
                    _data = cv2.copyMakeBorder(np.asarray(_data), top=1, bottom=1, left=1, right=1,
                                             borderType=cv2.BORDER_CONSTANT, value=[211, 211, 211])
                    recon = cv2.copyMakeBorder(np.asarray(recon), top=1, bottom=1, left=1, right=1,
                                             borderType=cv2.BORDER_CONSTANT, value=[211, 211, 211])
                    o_l = np.asarray(_data) if o_l == [] else np.concatenate((o_l, np.asarray(_data)), axis=1)
                    r_l = np.asarray(recon) if r_l == [] else np.concatenate((r_l, np.asarray(recon)), axis=1)
            img = cv2.cvtColor(np.float32(np.vstack((o_l, r_l)) * 255), cv2.COLOR_BGR2RGB)
            cv2.imwrite('{}/visuals/recon_epoch{}.png'.format(runPath, epoch),img)
        elif self.enc_name.lower() in ["txttransformer"]:
            recon_decoded, orig_decoded = output_onehot2text(recons_mat, list(data[0].squeeze().int()))
            output = open('{}/visuals/recon_{:03d}.txt'.format(runPath, epoch), "w")
            joined = []
            for o, r in zip(orig_decoded, recon_decoded):
                for l in [o, "|", r, "\n"]:
                    joined.append(l)
            output.writelines(["".join(joined)])
            output.close()
        elif self.enc_name.lower() == "audio":
             for i in range(3):
                numpy_to_wav(os.path.join(runPath,"visuals/",'orig_epoch{}_s{}.wav'.format(epoch, i)),
                         np.asarray(data[i].cpu()).astype(np.int16), 16000)
                numpy_to_wav(os.path.join(runPath,"visuals/",'recon_epoch{}_s{}.wav'.format(epoch, i)),
                         np.asarray(recons_mat[i].cpu()).astype(np.int16), 16000)

    def analyse(self, data, runPath, epoch, labels=None):
        """
        Plots KL divergences and T-SNE

        :param data: input data
        :type data: torch.tensor
        :param runPath: path to save data to
        :type runPath: str
        :param epoch: current epoch to name data
        :type epoch: int
        :param labels: labels for labelled T-SNE (optional)
        :type labels: list
        """
        zsl, kls_df = self.analyse_data(data, K=1, runPath=runPath, epoch=epoch, labels=labels)
        plot_kls_df(kls_df, '{}/visuals/kl_distance_{:03d}.png'.format(runPath, epoch))

    def generate_samples(self, N):
        """
        Generates samples from the latent space

        :param N: How many samples to make
        :type N: int
        :return: output reconstructions
        :rtype: torch.tensor
        """
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            if "transformer" in self.enc_name.lower():
                px_z = self.px_z(*self.dec([latents, None]))
            else:
                px_z = self.px_z(*self.dec(latents))
            data = get_mean(px_z)
        return data

    def reconstruct_data(self, data):
        """
        Reconstructs data

        :param data: modality on the input
        :type data: torch.tensor
        :return: reconstructed data
        :rtype: torch.tensor
        """
        self.eval()
        if self.enc_name not in ["Transformer", "TxtTransformer"]:
            with torch.no_grad():
                qz_x = self.qz_x(*self.enc(data))
                latents = qz_x.rsample()
                px_z = self.px_z(*self.dec(latents.unsqueeze(0)))
                recon = get_mean(px_z)
        else:
            with torch.no_grad():
                qz_x = self.qz_x(*self.enc(data))
                latents = qz_x.rsample()
                px_z = self.px_z(*self.dec([latents.unsqueeze(0), data[1]]))
                recon = get_mean(px_z)
        return recon

    def analyse_data(self, data, K, runPath, epoch, labels):
        """
        Makes latent samples and T-SNE

        :param data: test data
        :type data: list, torch.tensor
        :param K: make K samples from the posterior distribution
        :type K: int
        :param runPath: path where to save images
        :type runPath: str
        :param epoch: current epoch (to name images)
        :type epoch: int
        :param labels: labels for the data for labelled T-SNE (optional) - list of strings
        :type labels: list
        :return: returns K latent samples for each input
        :rtype: list
        """
        self.eval()
        with torch.no_grad():
            qz_x, _, zs = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, len(data)])).view(-1, pz.batch_shape[-1]),
                   zs.view(-1, zs.size(-1))]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        t_sne([x.detach().cpu() for x in zss[1:]], runPath, epoch, K, labels)
        return torch.cat(zsl, 0).cpu().numpy(), kls_df
