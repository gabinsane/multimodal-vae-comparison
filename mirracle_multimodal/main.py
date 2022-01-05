import argparse
import configparser
import json, yaml
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch import optim
import os
import models, objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data


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
    args = parser.parse_args()
    with open(args.cfg) as file: config = yaml.load(file)
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

model = "VAE" if len(mods) == 1 else config["mixing"].lower()
modelC = getattr(models, model)
params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods], [m["feature_dim"] for m in mods]]
if len(mods) == 1:
    params = [x[0] for x in params]
model = modelC(*params, config["n_latents"]).to(device)

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
with open('{}/config.json'.format(runPath), 'w') as fp:
    json.dump(config, fp)

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)
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
            data = unpack_data(dataT, device=device)
            d_len = data[0].shape[0]
        else:
            data = unpack_data(dataT[0], device=device)
            d_len = data.shape[0]
        optimizer.zero_grad()
        loss, kld, partial_l = objective(model, data, ltype=config["loss"])
        loss_m.append(loss/d_len)
        kld_m.append(kld/d_len)
        for i,l in enumerate(partial_l):
            partial_losses[i].append(l/d_len)
        loss.backward()
        optimizer.step()
        print("Training iteration {}/{}, loss: {}".format(it, int(len(train_loader.dataset)/config["batch_size"]), int(loss/d_len)))
    progress_d = {"Epoch": epoch, "Train Loss": get_loss_mean(loss_m), "Train KLD": get_loss_mean(kld_m)}
    for i, x in enumerate(partial_losses):
        progress_d["Train Mod_{}".format(i)] = get_loss_mean(x)
    lossmeter.update_train(progress_d)
    agg['train_loss'].append(get_loss_mean(loss_m))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def detach(listtorch):
    return [np.asarray(l.detach().cpu()) for l in listtorch]

def trest(epoch, agg, lossmeter):
    model.eval()
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(len(mods))]
    with torch.no_grad():
        for ix, dataT in enumerate(test_loader):
            if len(mods) > 1:
                data = unpack_data(dataT, device=device)
                d_len = data[0].shape[0]
            else:
                data = unpack_data(dataT[0], device=device)
                d_len = data.shape[0]
            loss, kld, partial_l = objective(model, data, ltype=config["loss"])
            loss_m.append(loss/d_len)
            kld_m.append(kld/d_len)
            for i, l in enumerate(partial_l):
                partial_losses[i].append(l/d_len)
            if ix == 0 and epoch % config["viz_freq"] == 0:
                model.reconstruct(data, runPath, epoch)
                model.generate(runPath, epoch)
                model.analyse(data, runPath, epoch)
    progress_d = {"Epoch": epoch, "Test Loss": get_loss_mean(loss_m), "Test KLD": get_loss_mean(kld_m)}
    for i, x in enumerate(partial_losses):
        progress_d["Test Mod_{}".format(i)] = get_loss_mean(x)
    lossmeter.update(progress_d)
    agg['test_loss'].append(get_loss_mean(loss_m))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))

def estimate_log_marginal(K):
    """Compute an IWAE estimate of the log-marginal likelihood of test data."""
    model.eval()
    marginal_loglik = 0
    with torch.no_grad():
        for dataT in test_loader:
            data = unpack_data(dataT, device=device)
            marginal_loglik += -objective(model, data, K).item()

    marginal_loglik /= len(test_loader.dataset)
    print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))

def get_loss_mean(loss):
    return round(float(torch.mean(torch.tensor(loss).detach().cpu())),3)

def load_data(pth,  imsize=64):
        import os, glob, numpy as np, imageio
        def generate(images):
            dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
            for i, image_path in enumerate(images[:3010]):
                image = imageio.imread(image_path)
                dataset[i, :] = image / 255
            return dataset.reshape(-1, 3, 64, 64)
        if any([os.path.isdir(x) for x in glob.glob(os.path.join(pth, "*"))]):
            subparts = (glob.glob(os.path.join(pth, "./*")))
            datasets = []
            for s in subparts:
                images = (glob.glob(os.path.join(s, "*.png")))
                d = generate(images)
                datasets.append(d)
            return datasets
        else:
            images = (glob.glob(os.path.join(pth, "*.png")))
            dataset = generate(images)
            return dataset

if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        conf = {"A":{"name":"Image"}, "B":{"name":"ImageTxt"}}
        lossmeter = Logger(runPath, mods)
        for epoch in range(1, int(config["epochs"]) + 1):
            train(epoch, agg, lossmeter)
            trest(epoch,agg, lossmeter)
            save_model(model, runPath + '/model.rar')
            if epoch % 100 == 0:
                save_model(model, runPath + '/model_epoch{}.rar'.format(epoch))
