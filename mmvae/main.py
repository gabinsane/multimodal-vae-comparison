import argparse
import sys
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch import optim
import os
import models
import objectives
import csv, yaml
from utils import Logger, Timer, save_model, save_vars, unpack_data

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--f', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--viz_freq', type=int, default=None,
                    help='frequency of viz savings')
parser.add_argument('--model', type=str, default=None, metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default=None, metavar='O',
                    help='objective to use (default: elbo)')
parser.add_argument('--loss', type=str, default=None, metavar='O',
                    help='loss to use (lprob/bce)')
parser.add_argument('--K', type=int, default=1, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=None, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=None, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--logp', action='store_true', default=False,
                    help='estimate tight marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=1, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-analytics', action='store_true', default=False,
                    help='disable plotting analytics')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1111, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--mod1', type=str, default=None)
parser.add_argument('--mod2', type=str, default=None)
parser.add_argument('--cfg', type=str, default="./config.yml", help="Path to config file.")
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

args.data_dim1, args.data_dim2 = [3,64,64], [3,64,64]
with open(args.cfg, "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
if not args.mod1:
    args.mod1 = config["modality_1"]["dataset"]
if ".pkl" in args.mod1:
    args.num_words1 = int(config["modality_1"]["num_words"])
    args.data_dim1 = int(os.path.basename(args.mod1.lower()).split("d")[0]) * args.num_words1 if "d" in args.mod1.lower() else 1
args.data1 = config["modality_1"]["type"]
if "modality_2" in config.keys():
    if not args.mod2:
        args.mod2 = config["modality_2"]["dataset"]
    if ".pkl" in args.mod2:
        args.num_words2 = int(config["modality_2"]["num_words"])
        args.data_dim2 = int(os.path.basename(args.mod2.lower()).split("d")[0]) * args.num_words2 if "d" in args.mod2.lower() else 1
    args.data2 = config["modality_2"]["type"]
if not args.epochs:
    args.epochs = int(config["general"]["n_epochs"])
if not args.latent_dim:
    args.latent_dim = int(config["general"]["n_latents"])
if not args.viz_freq:
    args.viz_freq = int(config["general"]["viz_freq"])
if not args.obj:
    args.obj = config["general"]["obj"]
if not args.loss:
    args.loss = config["general"]["loss"]
if not args.model:
    args.model = config["general"]["model"]
    if "poe" in args.obj:
        args.model = args.model.replace("2mods", "2mods_poe")
args.noisytxt = config["general"]["noisy_txt"]
# random seed
torch.backends.cudnn.benchmark = True
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# load args from disk if pretrained model path is given
pretrained_path = ""
if args.pre_trained:
    pretrained_path = args.pre_trained
    args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
if args.model == "uni":
    model = modelC(args, index=0).to(device)
else:
    model = modelC(args).to(device)

if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

if not args.f:
    args.f = model.modelName

# set up run path
runId = os.path.basename(args.cfg).split("config")[1].split(".")[0]

experiment_dir = os.path.join('results/', args.f)
if "src" in args.cfg:

    runPath = os.path.join(args.cfg.split("src/")[0],experiment_dir, runId)
else:
    runPath = os.path.join(experiment_dir, runId)
os.makedirs(runPath, exist_ok=True)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)
train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))
t_objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))

def train(epoch, agg, lossmeter):
    model.train()
    b_loss = 0
    loss_m = []
    kld_m = []
    img_loss_m = []
    txt_loss_m = []
    for i, dataT in enumerate(train_loader):
        if "2mods" in args.model:
            data = unpack_data(dataT, device=device)
        else:
            data = unpack_data(dataT[0], device=device)
        optimizer.zero_grad()
        loss, kld, img_loss, txt_loss = objective(model, data, K=args.K, ltype=args.loss)
        loss_m.append(loss)
        kld_m.append(kld)
        if "2mods" in args.model or args.data1 == "img":
            img_loss_m.append(img_loss)
            txt_loss_m.append(txt_loss)
        else:
            img_loss_m.append(txt_loss)
            txt_loss_m.append(img_loss)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    if "2mods" in args.model:
        progress_d = {"Epoch": epoch, "Train Loss": sum_det(loss_m)/len(train_loader.dataset),
                      "Train Image Loss": sum_det(img_loss_m)/len(train_loader.dataset),
                      "Train ImageTxt Loss": sum_det(txt_loss_m)/len(train_loader.dataset), "Train KLD": sum_det(kld_m) / len(train_loader.dataset)}
    else:
        if args.data1 == "txt":
            progress_d = {"Epoch": epoch, "Train Loss": sum_det(loss_m) / len(train_loader.dataset),
                          "Train ImageTxt Loss": sum_det(txt_loss_m) / len(train_loader.dataset)}
        else:
            progress_d = {"Epoch": epoch, "Train Loss": sum_det(loss_m) / len(train_loader.dataset),
                          "Train Image Loss": sum_det(img_loss_m) / len(train_loader.dataset)}

    lossmeter.update_train(progress_d)
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def test(epoch, agg, lossmeter):
    model.eval()
    b_loss = 0
    loss_m = []
    kld_m = []
    img_loss_m = []
    txt_loss_m = []
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            if "2mods" in args.model:
                data = unpack_data(dataT, device=device)
            else:
                data = unpack_data(dataT[0], device=device)
            loss, kld, img_loss, txt_loss = t_objective(model, data, K=args.K, ltype=args.loss)
            loss_m.append(loss)
            kld_m.append(kld)
            if "2mods" in args.model or args.data1 == "img":
                img_loss_m.append(img_loss)
                txt_loss_m.append(txt_loss)
            else:
                img_loss_m.append(txt_loss)
                txt_loss_m.append(img_loss)
            b_loss += loss.item()
            if i == 0 and epoch % args.viz_freq == 0:
                if (args.model =="uni" and "attrs.pkl" in args.mod1):
                    pass
                else:
                    model.reconstruct(data, runPath, epoch)
                    try:
                         model.generate(runPath, epoch)
                    except:
                        pass
                if not args.no_analytics:
                     model.analyse(data, runPath, epoch)
    if "2mods" in args.model:
        progress_d = {"Epoch": epoch, "Test Loss": sum_det(loss_m)/len(test_loader.dataset),
                      "Test Image Loss": sum_det(img_loss_m)/len(test_loader.dataset),
                      "Test ImageTxt Loss": sum_det(txt_loss_m)/len(test_loader.dataset), "Test KLD":sum_det(kld_m) / len(test_loader.dataset)}
    else:
        if args.data1 == "txt":
            progress_d = {"Epoch": epoch, "Test Loss": sum_det(loss_m)/len(test_loader.dataset),
                         "Test ImageTxt Loss": sum_det(txt_loss_m)/len(test_loader.dataset), "Test KLD":sum_det(kld_m) / len(test_loader.dataset)}
        else:
            progress_d = {"Epoch": epoch, "Test Loss": sum_det(loss_m) / len(test_loader.dataset),
                          "Test Image Loss": sum_det(img_loss_m) / len(test_loader.dataset),
                          "Test KLD": sum_det(kld_m) / len(test_loader.dataset)}
    lossmeter.update(progress_d)
    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))


def estimate_log_marginal(K):
    """Compute an IWAE estimate of the log-marginal likelihood of test data."""
    model.eval()
    marginal_loglik = 0
    with torch.no_grad():
        for dataT in test_loader:
            data = unpack_data(dataT, device=device)
            marginal_loglik += -t_objective(model, data, K).item()

    marginal_loglik /= len(test_loader.dataset)
    print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))

def sum_det(tn):
    if isinstance(tn[0], int):
        return 0
    else:
        return float(torch.stack(tn).sum().cpu().detach())

class Logger(object):
    """Saves training progress into csv"""
    def __init__(self, config, path):
        self.fields = ["Epoch", "Train Loss", "Test Loss", "Train Joint Loss", "Test Joint Loss", "Test KLD",
                       "Joint Mu", "Joint Logvar", "Train KLD"]
        self.path = path
        for key in config.keys():
            self.fields.append("Train {} Loss".format(config[key]["name"]))
            self.fields.append("Test {} Loss".format(config[key]["name"]))
            self.fields.append("Mu {}".format(config[key]["name"]))
            self.fields.append("Logvar {}".format(config[key]["name"]))
        self.reset()

    def reset(self):
        with open(os.path.join(self.path, "loss.csv"), mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writeheader()

    def update_train(self, val_d):
        self.dic = val_d

    def update(self, val_d):
        with open(os.path.join(self.path, "loss.csv"), mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writerow({**self.dic, **val_d})
        self.dic = {}

if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        conf = {"A":{"name":"Image"}, "B":{"name":"ImageTxt"}}
        lossmeter = Logger(conf, runPath)
        for epoch in range(1, args.epochs + 1):
            train(epoch, agg, lossmeter)
            test(epoch, agg, lossmeter)
            save_model(model, runPath + '/model.rar')
        if args.logp:  # compute as tight a marginal likelihood as possible
            estimate_log_marginal(5000)
