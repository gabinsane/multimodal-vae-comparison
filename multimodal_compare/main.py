import argparse
import yaml
import numpy as np
import torch
from models.trainer import Trainer


def parse_args():
    """
    Loads the .yml config specified in the --cfg argument. Any additional arguments override the values in the config.
    :return: dict; config
    :rtype: dict
    """
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
    args = parser.parse_args()
    with open(args.cfg) as file:
        config = yaml.safe_load(file)
    for name, value in vars(args).items():
        if value is not None and name != "cfg" and name in config.keys():
            config[name] = value
    return config


def main():
    config = parse_args()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config, dev)
    trainer.iterate_epochs()


if __name__ == '__main__':
    main()
