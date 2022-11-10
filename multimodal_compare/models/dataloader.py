from pytorch_lightning import LightningDataModule
import torch
import numpy as np
from models import datasets
from typing import Optional
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
        self.testpths = [x["test_datapath"] if "test_datapath" in x.keys() else None for x in self.config.mods]
        self.mod_types = [x["mod_type"] for x in self.config.mods]
        self.val_split = self.config.test_split
        self.dataset_name = self.config.dataset_name
        self.dataset_train = []
        self.dataset_val = []
        self.dataset_test = []
        self.datasets = self.prepare_data_classes()
        self.labels = self.config.labels
        self.labels_test = self.config.labels_test
        self.labels_val, self.labels_train = None, None
        self.batch_size = self.config.batch_size

    def get_dataset_class(self):
        """
        Get the dataset class object according to the dataset name

        :return: dataset class object
        :rtype: object
        """
        assert hasattr(datasets, self.dataset_name.upper()), "Did not find dataset with name {}".format(self.dataset_name.upper())
        return getattr(datasets, self.dataset_name.upper())

    def prepare_data_classes(self):
        datasets = []
        for i, p in enumerate(self.pths):
            datasets.append(self.get_dataset_class()(p, self.testpths[i], self.mod_types[i]))
        return datasets

    def setup(self, stage: Optional[str] = None) -> None:
        """ Loads appropriate dataset classes and makes data splits """
        if all([isinstance(x, TensorDataset) for x in [self.dataset_train, self.dataset_val]]):
            return
        shuffle = None
        for dataset in self.datasets:
            d = dataset.get_data()
            if shuffle is None:
                shuffle = np.random.permutation(len(d))
            d = d[shuffle]
            self.dataset_train.append(d[:int(len(d) * (1 - self.val_split))])
            self.dataset_val.append(d[int(len(d) * (1 - self.val_split)):])
        self.labels_train = list(np.asarray(self.get_labels())[shuffle])[:int(len(d) * (1 - self.val_split))] if self.get_labels() is not None else None
        self.labels_val = list(np.asarray(self.get_labels())[shuffle])[int(len(d) * (1 - self.val_split)):] if self.get_labels() is not None else None
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
        if len(batch.shape) == 3:
            dic = {"data": batch[:,:,:-1], "masks": batch[:,:,-1].bool()}
        elif len(batch.shape) == 4:
            dic = {"data": batch[:,:,:, :-1], "masks": batch[:,:,0,-1].squeeze().bool()}
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

        :param batch: input batch
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
                          num_workers=6)

    def val_dataloader(self) -> DataLoader:
        """Return Val DataLoader"""
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn,
                          num_workers=6)

    def predict_dataloader(self, batch_size, split="val") -> DataLoader:
        """Return Val DataLoader with custom batch size"""
        dataset = {"val":self.dataset_val, "test": self.dataset_test}[split]
        if split == "test":
            self.check_load_testdata()
            dataset = self.dataset_val if len(self.dataset_test) == 0 else dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn,
                              num_workers=0)

    def check_load_testdata(self):
        if len(self.dataset_test) == 0:
            shuffle = None
            self.dataset_test = []
            for dataset in self.datasets:
                d = dataset.get_test_data()
                if shuffle is None:
                    shuffle = np.random.permutation(len(d)) if d is not None else None
                if d is not None:
                    d = d[shuffle]
                    self.dataset_test.append(d[shuffle])
            self.labels_test = list(np.asarray(self.get_labels(split="test"))[shuffle]) if self.get_labels(
                split="test") is not None else None
            if len(self.dataset_train) == 1:
                self.dataset_test = TensorDataset(self.dataset_test[0]) if self.dataset_test[0] is not None else []
            else:
                self.dataset_test = TensorDataset(self.dataset_test) if len(self.dataset_test) > 0 else []

    def test_dataloader(self):
        """Return Test DataLoader"""
        self.check_load_testdata()
        if len(self.dataset_test) > 0:
            return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn,
                              num_workers=6)
        else:
            print("Note: final testing with validation dataloader because test data was not provided.")
            return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=self.collate_fn,
                              num_workers=0)

    def get_labels(self, split="all"):
        """
        Return data labels for given indices if available

        :param split: "all"/"train"/"val/test" depending on the data split
        :type split: str
        :return: list of labels for given indices
        :rtype: list
        """
        labels = {"all": self.labels, "val": self.labels_val, "train": self.labels_train, "test":self.labels_test}[split]
        if labels is not None:
            return labels
        else:
            for d in self.datasets:
                if hasattr(d, "labels") and d.labels() is not None:
                    labels = d.get_labels(split="train")
                    if split == "all":
                        return labels
                    elif split == "val":
                        return labels[int(len(labels) * (1 - self.val_split)):]
                    elif split == "train":
                        return labels[:int(len(labels) * (1 - self.val_split))]
                    elif split == "test":
                        l = d.get_labels(split="test")
                        if l is not None:
                            return d.get_labels(split="test")
                        else:
                            return labels[int(len(labels) * (1 - self.val_split)):]
        return None

    def get_label_for_indices(self, indices, split):
        """Get labels for the given data split according to indices"""
        return [self.get_labels(split)[x] for x in indices] if self.get_labels(split) is not None else None

    def get_num_samples(self, num_samples, split="val"):
        """Returns batch of the predict_dataloader together with the indices"""
        while True:
            try:
                data = next(iter(self.predict_dataloader(num_samples, split=split)))
                index = iter(self.predict_dataloader(num_samples, split=split))._next_index()
                labels = self.get_label_for_indices(index, split)
                return data, labels
            except:
                continue
