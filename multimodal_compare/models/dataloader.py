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
            self.dataset_train = self.dataset_train[0]
            self.dataset_val = self.dataset_val[0]
        else:
            self.dataset_train = TensorDataset(self.dataset_train)
            self.dataset_val = TensorDataset(self.dataset_val)

    def make_masks(self, batch):
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in batch])))
        data = list(torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0))
        dic = {"data": torch.stack(data), "masks": masks}
        return dic

    def prepare_singlemodal(self, batch, mod_index):
        d = {}
        if self.datasets[mod_index-1].has_masks:
            d["mod_{}".format(mod_index)] = self.make_masks(batch)
        else:
            d["mod_{}".format(mod_index)] = {"data": batch, "masks":None}
        return d

    def collate_fn(self, batch):
        b_dict = {}
        if len(self.config.mods) > 1:
            for i in range(len(self.config.mods)):
                modality = [x[i] for x in batch]
                b_dict.update(self.prepare_singlemodal(torch.stack(modality), i+1))
        else:
            b_dict.update(self.prepare_singlemodal(batch, 1))
        return b_dict


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.collate_fn,
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.collate_fn,
                          )
