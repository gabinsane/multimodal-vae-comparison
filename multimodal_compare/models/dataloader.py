from pytorch_lightning import LightningDataModule
import torch, os
from models import datasets
from typing import Optional
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
        self.batch_size = self.config.batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """ Loads appropriate dataset classes and makes data splits """
        datasets_full = []
        for i, p in enumerate(self.pths):
                 assert hasattr(datasets, self.dataset_name.upper()), "Did not find dataset with name {}".format(self.dataset_name)
                 datasets_full.append(getattr(datasets, self.dataset_name.upper())(p, self.mod_types[i]))
        for dataset in datasets_full:
            d = dataset.get_data()
            self.dataset_train.append(d[:int(len(d) * (1 - self.val_split))])
            self.dataset_val.append(d[int(len(d) * (1 - self.val_split)):])
        if len(self.dataset_train) == 1:
            self.dataset_train = self.dataset_train[0]
            self.dataset_val = self.dataset_val[0]
        else:
            self.dataset_train = torch.stack(self.dataset_train)
            self.dataset_val = torch.stack(self.dataset_val)

    def collate_fn(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                           num_workers=os.cpu_count())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                          num_workers=os.cpu_count())
