import argparse
import os
import pytorch_lightning as pl
from models.trainer import MultimodalVAE
from models.config_cls import Config
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from models.dataloader import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
from pickle import dumps

def identity(string):
    return string

parser = argparse.ArgumentParser()
parser.register('type', None, identity)
_ = dumps(parser)
parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
parser.add_argument('-p', '--precision', type=str, default=32,
                    help='Value for mixed precision training. Allowed values: 64, 32, 16, bf16')
parser.add_argument('--viz_freq', type=int, default=None,
                    help='frequency of visualization savings (number of iterations)')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Size of the training batch')
parser.add_argument('--obj', type=str, metavar='O', default=None,
                    help='objective to use (moe_elbo/poe_elbo_semi)')
parser.add_argument('--loss', type=str, metavar='O', default=None,
                    help='loss to use (lprob/bce)')
parser.add_argument('--n_latents', type=int, default=None,
                    help='latent vector dimensionality')
parser.add_argument('--pre_trained', type=str, default=None,
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--seed', type=int, metavar='S', default=None,
                    help='seed number')
parser.add_argument('--exp_name', type=str, default=None,
                    help='name of folder')
parser.add_argument('--optimizer', type=str, default=None,
                    help='optimizer')

def main(config):
    pl.seed_everything(config.seed)
    data_module = DataModule(config)
    model_wrapped = MultimodalVAE(config, data_module.get_dataset_class().feature_dims)
    profiler = SimpleProfiler(dirpath=os.path.join(config.mPath, "model"), filename="profiler_output")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config.mPath, "model"), save_last=True, save_top_k=1, mode="min")
    logger2 = CSVLogger(save_dir=config.mPath, name="metrics", flush_logs_every_n_steps=1, version="csv")
    logger1 = TensorBoardLogger(config.mPath, name="metrics", log_graph=True, version="tensorboard")
    trainer_kwargs = {"profiler": profiler, "accelerator":"gpu",
                      "default_root_dir": config.mPath, "max_epochs": config.epochs, "check_val_every_n_epoch": 1,
                      "callbacks": [checkpoint_callback], "logger":[logger1, logger2], "precision":args.precision, "devices":-1}
    pl_trainer = pl.Trainer(**trainer_kwargs)
    pl_trainer.fit(model_wrapped, datamodule=data_module)
    pl_trainer.test(ckpt_path="best", datamodule=data_module)

if __name__ == '__main__':
    args = parser.parse_args()
    config = Config(parser)
    if config.iterseeds > 1: # iterate over number of seeds defined in iterseeds
        for seed in range(config.iterseeds):
            if seed > 0: # after first training we need to make new path for the new model
                config = Config(parser)
            config.change_seed(config.seed+seed)
            config.dump_config()
            main(config)
    else:
        main(config)
