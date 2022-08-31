from pytorch_lightning import LightningDataModule
import torch, os
from models import datasets
from utils import lengths_to_mask
from typing import Optional
import numpy as np
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self, config):
        """
        Class for dataset loading and adjustments for training

        :param pth: parsed config
        :type pth: object
        """
        super().__init__()
        self.config = config
        self.pths = [x["path"] for x in self.config.mods]
        self.mod_types = [x["mod_type"] for x in self.config.mods]
        self.val_split = self.config.test_split
        self.dataset_name = self.config.dataset_name
        self.dataset_train = []
        self.dataset_val = []
        self.datasets = []
        self.batch_size = self.config.batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """ Loads appropriate dataset classes and makes data splits """
        for i, p in enumerate(self.pths):
                 assert hasattr(datasets, self.dataset_name.upper()), "Did not find dataset with name {}".format(self.dataset_name)
                 self.datasets.append(getattr(datasets, self.dataset_name.upper())(p, self.mod_types[i]))
        for dataset in self.datasets:
            d = dataset.get_data()
            self.dataset_train.append(d[:int(len(d) * (1 - self.val_split))])
            self.dataset_val.append(d[int(len(d) * (1 - self.val_split)):])
        if len(self.dataset_train) == 1:
            self.dataset_train = TensorDataset(self.dataset_train[0])
            self.dataset_val = TensorDataset(self.dataset_val[0])
        else:
            self.dataset_train = TensorDataset(self.dataset_train)
            self.dataset_val = TensorDataset(self.dataset_val)

    def make_masks(self, batch):
        """
        Makes masks for sequential data

        :param batch: data batch
        :type batch: torch.tensor
        :return: dictionary with data and masks
        :rtype: dict
        """
        dic = {"data": batch[:,:,:-1], "masks": batch[:,:,-1].bool()}
        return dic

    def prepare_singlemodal(self, batch, mod_index):
        """
        Prepares singlemodal data for given modality

        :param batch: input batch
        :type batch: list
        :param mod_index: index of the modality (as the order in config)
        :type mod_index: int
        :return: prepared data for training
        :rtype: dict
        """
        d = {}
        if self.datasets[mod_index-1].has_masks:
            d["mod_{}".format(mod_index)] = self.make_masks(batch)
        else:
            d["mod_{}".format(mod_index)] = {"data": batch, "masks":None}
        d["mod_{}".format(mod_index)]["categorical"] = self.datasets[mod_index-1].categorical
        return d

    def collate_fn(self, batch):
        """
        Custom collate function that puts data in a dictionary and prepares masks if needed

        :param batch: input batcg
        :type batch: list
        :return: dictionary with data batch
        :rtype: dict
        """
        b_dict = {}
        if len(self.config.mods) > 1:
            for i in range(len(self.config.mods)):
                modality = [x[i] for x in batch]
                b_dict.update(self.prepare_singlemodal(torch.stack(modality), i+1))
        else:
            b_dict.update(self.prepare_singlemodal(torch.stack(batch), 1))
        return b_dict


    def train_dataloader(self) -> DataLoader:
        """Return Train DataLoader"""
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn,
                          num_workers=0)

    def val_dataloader(self) -> DataLoader:
        """Return Val DataLoader"""
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn,
                          num_workers=0)

    def predict_dataloader(self, batch_size) -> DataLoader:
        """Return Val DataLoader with custom batch size"""
        return DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn,
                          num_workers=0)
