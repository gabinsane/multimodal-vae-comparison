from models.trainer import MultimodalVAE
from models.config_cls import Config
import os, argparse, torch
from models.dataloader import DataModule
import pytorch_lightning as pl
pl.seed_everything(0)

class MultimodalVAEInfer():
    """
    Class that includes methods for direct model testing and evaluation.
    The users can use this class to test their trained model with their own data (outside Dataloaders), compare multiple models etc.
    """
    def __init__(self, model):
        """
        :param model: path to the .ckpt file of the trained model
        :type model: str
        """
        self.base_path = None
        self.base_path = self.get_base_path(model)
        self.path = model
        assert os.path.exists(self.path), "Provided path does not exist!"
        assert self.path.split(".")[-1] == "ckpt", "Provided path is not a .ckpt file!"
        self.config = self.get_config()
        self.datamod = self.get_datamodule()
        self.model = MultimodalVAE(self.config, self.datamod.get_dataset_class().feature_dims)
        self.model = MultimodalVAE.load_from_checkpoint(self.path, cfg=self.config, feature_dims=self.datamod.get_dataset_class().feature_dims)
        self.model.model = self.model.model.eval().to(torch.device("cuda"))

    def get_base_path(self, path):
        """Finds the base directory of the model"""
        if self.base_path is not None:
            return self.base_path
        else:
            if not os.path.exists(path):
                raise ValueError(f"Path is not valid: {path}")
        while "model" in path:
            path = os.path.dirname(path)
        assert os.path.exists(path)
        return path

    def get_config(self):
        """Creates the Config instance based on the provided path"""
        cfg_path = os.path.join(self.base_path, "config.yml")
        cfg = Config(cfg_path)
        return cfg

    def get_datamodule(self, load_data=True):
        """Creates an instance of the DataModule class. Necessary for accessing the specific data processing tools"""
        data_module = DataModule(self.config)
        if load_data:
            data_module.setup()
        return data_module

    def make_dataloaders(self):
        """Loads the train, val (and test, if available) dataloaders within the datamodule class"""
        self.datamod.setup()

    def eval_statistics(self):
        """Runs the official evaluation routine defined in trainer.py. If applicable, calculates statistics"""
        trainer = pl.Trainer(accelerator="gpu")
        if len(self.datamod.dataset_val) == 0:
            self.make_dataloaders()
        trainer.test(self.model, datamodule=self.datamod)

    def get_wrapped_model(self):
        """Returns the Trainer class with loaded Datamodule outside Pytorch Lightning"""
        if len(self.datamod.dataset_val) == 0:
            self.make_dataloaders()
        model = self.model.to(torch.device("cuda"))
        model.datamodule = self.datamod
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mpath", type=str, help="path to the .ckpt model file. Relative or absolute")
    args = parser.parse_args()
    cl = MultimodalVAEInfer(args.mpath)
    cl.eval_statistics()
