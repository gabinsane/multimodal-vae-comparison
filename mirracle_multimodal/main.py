import argparse
import sys
import configparser
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


def parse_args():
    conf_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                          add_help=False)
    conf_parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    if args.cfg:
        conf = configparser.SafeConfigParser()
        conf.read([args.cfg])
        defaults = dict(conf.items("general"))
        defaults["cfg"] = args.cfg
    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**defaults)
    parser.add_argument('--viz_freq', type=int,
                        help='frequency of visualization savings (number of iterations)')
    parser.add_argument('--batch_size', type=int,
                        help='Size of the training batch')
    parser.add_argument('--modalities_num', type=int,
                        help='number of modalities to train on')
    parser.add_argument('--obj', type=str, metavar='O',
                        help='objective to use (moe_elbo/poe_elbo_semi)')
    parser.add_argument('--loss', type=str, metavar='O',
                        help='loss to use (lprob/bce)')
    parser.add_argument('--llik_scaling', type=float,
                        help='likelihood scaling for reconstruction loss'
                             ', set as 0 to use balance the mods and 1 to not')
    parser.add_argument('--n_latents', type=int,
                        help='latent vector dimensionality')
    parser.add_argument('--pre_trained', type=str,
                        help='path to pre-trained model (train from scratch if empty)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disable CUDA usage')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='seed number')
    parser.add_argument('--exp_name', type=str,
                        help='name of folder')
    args = parser.parse_args(remaining_argv)
    return args

args = parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

model = str(args.modalities_num) if args.modalities_num == 1 else "_".join((str(args.modalities_num), args.mixing))
# load model
modelC = getattr(models, 'VAE_{}'.format(model))
model = modelC(vars(args)).to(device)

if args.pre_trained:
    print('Loading model {} from {}'.format(model.modelName, args.pre_trained))
    model.load_state_dict(torch.load(args.pre_trained + '/model.rar'))
    model._pz_params = model._pz_params

# set up run path
runPath = os.path.join('results/', args.exp_name)
os.makedirs(runPath, exist_ok=True)
print('Expt:', runPath)

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
                    + ("_".join((args.obj, args.mixing)) if hasattr(model, 'vaes') else args.obj))

def train(epoch, agg, lossmeter):
    model.train()
    loss_m = []
    kld_m = []
    partial_losses =  [[] for _ in range(args.modalities_num)]
    for it, dataT in enumerate(train_loader):
        if int(args.modalities_num) > 1:
            data = unpack_data(dataT, device=device)
            d_len = data[0].shape[0]
        else:
            data = unpack_data(dataT[0], device=device)
            d_len = data.shape[0]
        optimizer.zero_grad()
        loss, kld, partial_l = objective(model, data, K=1, ltype=args.loss)
        loss_m.append(loss/d_len)
        kld_m.append(kld/d_len)
        for i,l in enumerate(partial_l):
            partial_losses[i].append(l/d_len)
        loss.backward()
        optimizer.step()
        print("Training iteration {}/{}, loss: {}".format(it, len(train_loader.dataset)/args.batch_size, int(loss/d_len)))
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
    partial_losses =  [[] for _ in range(args.modalities_num)]
    with torch.no_grad():
        for ix, dataT in enumerate(test_loader):
            if int(args.modalities_num) > 1:
                data = unpack_data(dataT, device=device)
                d_len = data[0].shape[0]
            else:
                data = unpack_data(dataT[0], device=device)
                d_len = data.shape[0]
            loss, kld, partial_l = objective(model, data, K=1, ltype=args.loss)
            loss_m.append(loss/d_len)
            kld_m.append(kld/d_len)
            for i, l in enumerate(partial_l):
                partial_losses[i].append(l/d_len)
            if ix == 0 and epoch % args.viz_freq == 0:
                model.reconstruct(data, runPath, epoch)
                model.generate(runPath, epoch)
                model.analyse(data, runPath, epoch)
    progress_d = {"Epoch": epoch, "Test Loss": get_loss_mean(loss_m), "Test KLD": get_loss_mean(kld_m)}
    for i, x in enumerate(partial_losses):
        progress_d["Test Mod_{}".format(i)] = get_loss_mean(x)
    lossmeter.update(progress_d)
    agg['test_loss'].append(get_loss_mean(loss_m))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))

# def vaetest(epoch, agg, lossmeter):
#     model.eval()
#     b_loss = 0
#     loss_m = []
#     kld_m = []
#     mod1_loss_m = []
#     mod2_loss_m = []
#     with torch.no_grad():
#         for i, dataT in enumerate(test_loader):
#             if int(args.modalities_num) > 1:
#                 data = unpack_data(dataT, device=device)
#             else:
#                 data = unpack_data(dataT[0], device=device)
#             loss, kld, mod1_loss, mod2_loss = objective(model, data, K=1, ltype=args.loss)
#             loss_m.append(loss)
#             kld_m.append(kld)
#             mod1_loss_m.append(mod1_loss)
#             mod2_loss_m.append(mod2_loss)
#             b_loss += loss.item()
#             if i == 0 and epoch % args.viz_freq == 0:
#                 if (int(args.modalities_num) == 1 and "attrs.pkl" in args.mod_path):
#                     pass
#                 else:
#                     model.reconstruct(data, runPath, epoch)
#                     try:
#                          model.generate(runPath, epoch)
#                     except:
#                         pass
#                     model.analyse(data, runPath, epoch)
#     if int(args.modalities_num) > 1:
#         progress_d = {"Epoch": epoch, "Test Loss": sum_det(loss_m)/len(test_loader.dataset),
#                       "Test {}1 Loss".format(args.mod1_type.upper()): sum_det(mod1_loss_m)/len(test_loader.dataset),
#                       "Test {}2 Loss".format(args.mod2_type.upper()): sum_det(mod2_loss_m)//len(test_loader.dataset), "Test KLD":sum_det(kld_m) / len(test_loader.dataset)}
#     else:
#           progress_d = {"Epoch": epoch, "Test Loss": np.mean(detach(loss_m)),
#                          "Test {} Loss".format(args.mod_type.upper()): np.mean(detach(mod1_loss_m)), "Test KLD": np.mean(detach(kld_m))}
#
#     lossmeter.update(progress_d)
#     agg['test_loss'].append(b_loss / len(test_loader.dataset))
#     print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))


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
        lossmeter = Logger(runPath, args)
        for epoch in range(1, int(args.epochs) + 1):
            train(epoch, agg, lossmeter)
            trest(epoch,agg, lossmeter)
            save_model(model, runPath + '/model.rar')
            if epoch % 100 == 0:
                save_model(model, runPath + '/model_epoch{}.rar'.format(epoch))
