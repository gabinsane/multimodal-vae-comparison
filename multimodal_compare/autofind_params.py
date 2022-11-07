import argparse
import numpy as np
import torch, os
import pytorch_lightning as pl
from models.trainer import MultimodalVAE
from models.config_cls import Config
from models.dataloader import DataModule
from pytorch_lightning.callbacks import StochasticWeightAveraging

def identity(string):
    return string

def find_lr(trainer, model, data_module, config):
    lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
    fig = lr_finder.plot(suggest=True)
    fig.savefig(os.path.join(config.mPath, "learning_rate_search.png"))
    fig.show()
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    # update hparams of the model
    print("Suggeed learning rate for this config is {}".format(new_lr))
    return new_lr

def find_max_batch_size(trainer, model, data_module):
    new_batch_size = trainer.tuner.scale_batch_size(model, datamodule=data_module)
    print("Maximum batch size for this config and machine is {}".format(new_batch_size))
    return new_batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', None, identity)
    parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
    config = Config(parser)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    data_module = DataModule(config)
    model_wrapped = MultimodalVAE(config, data_module.get_dataset_class().feature_dims)
    trainer_kwargs = {"accelerator": "gpu",
                      "default_root_dir": config.mPath, "callbacks": [StochasticWeightAveraging(swa_lrs=1e-2)]}
    pl_trainer = pl.Trainer(**trainer_kwargs)
    find_max_batch_size(pl_trainer, model_wrapped, data_module)
    find_lr(pl_trainer, model_wrapped, data_module, config)
