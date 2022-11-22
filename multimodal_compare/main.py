import argparse
import numpy as np
import torch
import yaml, os
import pytorch_lightning as pl
from models.trainer import MultimodalVAE
from models.config_cls import Config
from models.dataloader import DataModule
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
parser.add_argument('--viz_freq', type=int, default=None,
                    help='frequency of visualization savings (number of iterations)')
parser.add_argument('--compare', action='store_true', help='Whether to perform grid search')
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

def main():
    config = Config(parser)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    data_module = DataModule(config)
    model_wrapped = MultimodalVAE(config, data_module.get_dataset_class().feature_dims)
    profiler = SimpleProfiler(dirpath=config.mPath, filename="profiler_output")
    #logger = TensorBoardLogger("lightning_logs", name="VAEmodel", log_graph=True)
    trainer_kwargs = {"profiler": profiler, "accelerator":"gpu",
                      "default_root_dir": config.mPath, "max_epochs": config.epochs, "check_val_every_n_epoch": 1,
                      "callbacks": [StochasticWeightAveraging(swa_lrs=1e-2)]}
    pl_trainer = pl.Trainer(**trainer_kwargs)
    pl_trainer.fit(model_wrapped, datamodule=data_module)
    pl_trainer.test(ckpt_path="best", datamodule=data_module)

def identity(string):
    return string

def main_tune(config):
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data_module = DataModule(Config(config))
    tunecallback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
    model_wrapped = MultimodalVAE(Config(config), data_module.get_dataset_class().feature_dims)
    profiler = SimpleProfiler(dirpath=os.path.join('results/', config["exp_name"]), filename="profiler_output")
    logger = TensorBoardLogger("lightning_logs", name="VAEmodel", log_graph=True)
    trainer_kwargs = {"profiler": profiler, "logger": logger, "accelerator":"gpu",
                      "default_root_dir": os.path.join('results/', config["exp_name"]), "max_epochs": config["epochs"], "check_val_every_n_epoch": 1,
                      "callbacks": [EarlyStopping(monitor="val_loss", mode="min"),
                                    StochasticWeightAveraging(swa_lrs=1e-2), tunecallback]}
    pl_trainer = pl.Trainer(**trainer_kwargs)
    pl_trainer.fit(model_wrapped, datamodule=data_module)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.compare:
        with open(args.cfg) as file:
            config = yaml.safe_load(file)
        config["n_latents"] = tune.choice([32, 64, 128])

        tuner = tune.Tuner(tune.with_resources(main_tune, {"cpu": 6, "gpu":1}),
             tune_config=tune.TuneConfig(metric="loss", mode="min",),
             param_space=config,
             run_config=air.RunConfig(name="tune_vae"),
         )
        num_epochs = 10

        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=1,
            reduction_factor=2)

        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)
    else:
        main()
