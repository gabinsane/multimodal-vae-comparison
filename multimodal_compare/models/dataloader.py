import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
from utils import get_path_type, load_images, one_hot_encode, check_img_normalize


def load_dataset(model, batch_size, device='cuda'):
    """
    Loads the training and testing dataset

    :param model: the model we are loading data for
    :type model: TorchMMVAE
    :param batch_size: batch size
    :type batch_size: int
    :param device: device to put data on
    :type device: str
    :return: train and test datasets
    :rtype: tuple(torch.DataLoader, torch.DataLoader)
    """
    trains, tests = [], []
    for x in range(len(model.vaes)):
        t, v = model.vaes[x].load_dataset(batch_size, device)
        trains.append(t.dataset)
        tests.append(v.dataset)
    train_data = TensorDataset(trains)
    test_data = TensorDataset(tests)
    kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
    if any(["transformer" in e.lower() for e in model.encoders]): kwargs["collate_fn"] = model.seq_collate_fn
    train = DataLoader(train_data, batch_size=batch_size, shuffle=False, **kwargs)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    return train, test


def load_data(pth, data_dim, network_type, mod_type):
    """
    Loads the data from path

    :return: data prepared for training
    :rtype: torch.tensor
    """
    dtype = get_path_type(pth)
    if dtype == "dir":
        d = load_images(pth, data_dim)
    elif dtype == "torch":
        d = torch.load(pth)
    elif dtype == "pickle":
        with open(pth, 'rb') as handle:
            d = pickle.load(handle)
    d = prepare_for_encoder(d, network_type, mod_type, pth)
    return d

def prepare_for_encoder(data, network_type, mod_type, pth):
    """
    Prepares the data for training.

    :param data: the loaded data
    :type data: Union[list, torch.tensor, ndarray]
    :return: data reshaped for training,
    :rtype: torch.tensor
    """
    if network_type.lower() in ["transformer", "cnn", "3dcnn", "fnn"]:
        data = [torch.from_numpy(np.asarray(x).astype(np.float)) for x in data]
        if network_type in ["cnn", "fnn"]:
            data = torch.stack(data).transpose(1, 3)
        if "transformer" in network_type.lower():
            if len(data[0].shape) < 3:
                data = [torch.unsqueeze(i, dim=1) for i in data]
    elif "text" in mod_type:
        if len(data[0]) > 1 and not isinstance(data[0], str):
            data = [" ".join(x) for x in data] if not "cub_" in pth else data
        data = [one_hot_encode(len(f), f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        if "transformer" not in network_type.lower():
            data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
    if network_type.lower() == "audioconv":
        prepare_audio(data)
    if "image" in mod_type:
        data = check_img_normalize(data)
    return data

def prepare_audio(data):
    """
    @TODO Support for audio data

    :param data: input audio sequences
    :type data: list
    :return: padded sequences for training
    :rtype: torch.tensor
    """

    d = [torch.from_numpy(np.asarray(x).astype(np.int16)) for x in data]
    return torch.nn.utils.rnn.pad_packed_sequence(d, batch_first=True, padding_value=0.0)