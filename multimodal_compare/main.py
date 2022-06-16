import argparse
import json, yaml
from collections import defaultdict
import pickle
import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from torch import optim
import os
from eval.infer import plot_loss, eval_reconstruct, eval_sample
import models
from models import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, pad_seq_data


def parse_args():
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
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disable CUDA usage')
    parser.add_argument('--seed', type=int, metavar='S', default=None,
                        help='seed number')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='name of folder')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='optimizer')
    args = parser.parse_args()
    with open(args.cfg) as file: config = yaml.safe_load(file)
    for name, value in vars(args).items():
        if value is not None and name != "cfg" and name in config.keys():
            config[name] = value
    modalities = []
    for x in range(20):
        if "modality_{}".format(x) in list(config.keys()):
            modalities.append(config["modality_{}".format(x)])
    return config, modalities, args

config, mods, args = parse_args()
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
np.random.seed(config["seed"])
torch.backends.cudnn.benchmark = True

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

labels = None
if config["labels"]:
    with open(config["labels"], 'rb') as handle:
        labels = pickle.load(handle)

model = "VAE" if len(mods) == 1 else config["mixing"].lower()
modelC = getattr(models, model)
params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods], [m["feature_dim"] for m in mods]]
if len(mods) == 1:
    params = [x[0] for x in params]
model = modelC(*params, config["n_latents"], config["batch_size"]).to(device)

if config["pre_trained"]:
    print('Loading model {} from {}'.format(model.modelName, config["pre_trained"]))
    model.load_state_dict(torch.load(config["pre_trained"] + '/model.rar'))
    model._pz_params = model._pz_params

# set up run path
runPath = os.path.join('results/', config["exp_name"])
os.makedirs(runPath, exist_ok=True)
os.makedirs(os.path.join(runPath, "visuals"), exist_ok=True)
print('Expt:', runPath)

# save args to run
with open('{}/config.json'.format(runPath), 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

# preparation for training
if config["optimizer"].lower() == "adam":
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["lr"]), amsgrad=True)
elif config["optimizer"].lower() == "adabelief":
    optimizer = AdaBelief(model.parameters(), lr=float(config["lr"]), eps=1e-16, betas=(0.9,0.999), weight_decouple=True, rectify=False, print_change_log=False)

train_loader, test_loader = model.getDataLoaders(config["batch_size"], device=device)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + ("_".join((config["obj"], config["mixing"])) if hasattr(model, 'vaes') else config["obj"]))

def train(epoch, agg, lossmeter):
    model.train()
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(len(mods))]
    for it, dataT in enumerate(train_loader):
        if len(mods) > 1:
            if not isinstance(dataT, tuple):
                data = unpack_data(dataT, device=device)
                d_len = len(data[0])
            else:
                data, masks = dataT
                data = pad_seq_data(data, masks)
                d_len = len(data[0])
        else:
            if "transformer" in config["modality_1"]["encoder"].lower():
                data, masks = dataT
                data = [data.to(device), masks]
                d_len = len(data[0])
            else:
                data = unpack_data(dataT[0], device=device)
                d_len = len(data)
        optimizer.zero_grad()
        loss, kld, partial_l = objective(model, data, ltype=config["loss"])
        loss_m.append(loss)
        kld_m.append(kld)
        for i,l in enumerate(partial_l):
            partial_losses[i].append(l)
        loss.backward()
        optimizer.step()
        print("Training iteration {}/{}, loss: {}".format(it, int(len(train_loader.dataset)/config["batch_size"]), int(loss)))
    progress_d = {"Epoch": epoch, "Train Loss": get_loss_mean(loss_m), "Train KLD": get_loss_mean(kld_m)}
    for i, x in enumerate(partial_losses):
        progress_d["Train Mod_{}".format(i)] = get_loss_mean(x)
    lossmeter.update_train(progress_d)
    agg['train_loss'].append(get_loss_mean(loss_m))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def detach(listtorch):
    return [np.asarray(l.detach().cpu()) for l in listtorch]

def test(epoch, agg, lossmeter):
    model.eval()
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(len(mods))]
    with torch.no_grad():
        for ix, dataT in enumerate(test_loader):
            if len(mods) > 1:
                if not isinstance(dataT, tuple):
                    data = unpack_data(dataT, device=device)
                    d_len = len(data[0])
                else:
                    data, masks = dataT
                    data = pad_seq_data(data, masks)
                    d_len = len(data[0])
            else:
                if "transformer" in config["modality_1"]["encoder"].lower():
                    data, masks = dataT
                    data = [data.to(device), masks]
                    d_len = len(data[0])
                else:
                    data = unpack_data(dataT[0], device=device)
                    d_len = len(data)
            loss, kld, partial_l = objective(model, data, ltype=config["loss"])
            loss_m.append(loss)
            kld_m.append(kld)
            for i, l in enumerate(partial_l):
                partial_losses[i].append(l)
            if ix == 0 and epoch % config["viz_freq"] == 0:
                model.reconstruct(data, runPath, epoch)
                model.generate(runPath, epoch)
                if labels:
                     model.analyse(data, runPath, epoch, labels[int(len(labels)*0.9):int(len(labels)*0.9)+d_len])
                else:
                     model.analyse(data, runPath, epoch)
    progress_d = {"Epoch": epoch, "Test Loss": get_loss_mean(loss_m), "Test KLD": get_loss_mean(kld_m)}
    for i, x in enumerate(partial_losses):
        progress_d["Test Mod_{}".format(i)] = get_loss_mean(x)
    lossmeter.update(progress_d)
    agg['test_loss'].append(get_loss_mean(loss_m))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))

def get_loss_mean(loss):
    return round(float(torch.mean(torch.tensor(loss).detach().cpu())),3)


if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        lossmeter = Logger(runPath, mods)
        for epoch in range(1, int(config["epochs"]) + 1):
            train(epoch, agg, lossmeter)
            test(epoch,agg, lossmeter)
            save_model(model, runPath + '/model.rar')
        plot_loss(runPath)
        eval_sample(runPath)
        eval_reconstruct(runPath)
