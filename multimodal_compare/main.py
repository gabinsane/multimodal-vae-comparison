import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from models.trainer import Trainer
from models.config import Config
from models.dataloader import DataModule

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
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

def main():
    config = Config(parser)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    model_wrapped = Trainer(config)
    pl_trainer = pl.Trainer(gpus=1, default_root_dir=config.mPath, max_epochs=config.epochs)
    data_module = DataModule(config)
    pl_trainer.fit(model_wrapped, data_module)

if __name__ == '__main__':
    main()
