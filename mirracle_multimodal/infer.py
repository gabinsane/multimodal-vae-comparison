import torch, numpy as np
import models, pickle
from utils import unpack_data
from PIL import Image, ImageDraw, ImageFont,  ImageOps
import cv2, os
from glob import glob
import pandas as pd
import math, statistics
import matplotlib.pyplot as plt
import json
import argparse
import csv
import glob
from vis import t_sne

def parse_args(cfg_path=None):
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
    with open(os.path.join(cfg_path, "config.json")) as file: config = json.load(file)
    for name, value in vars(args).items():
        if value is not None and name != "cfg" and name in config.keys():
            config[name] = value
    modalities = []
    for x in range(20):
        if "modality_{}".format(x) in list(config.keys()):
            modalities.append(config["modality_{}".format(x)])
    return config, modalities, args

fonts = ["/usr/share/fonts/truetype/freefont/FreeSans.ttf"]
VECDIM = 4096
plot_colors = ["blue", "green", "red", "cyan", "magenta", "orange", "navy", "maroon", "brown"]

def eval(path, trainloader=None, testloader=None):
    model, args, trainloader, testloader = load_model(path, batch=16)
    device = torch.device("cuda" if args.cuda else "cpu")
    for i, data in enumerate(testloader):
        if "2mods" in args.model:
            d = unpack_data(data, device=device)
        else:
            d = unpack_data(data[0], device=device)
        model.reconstruct(d, path, "eval_rec{}".format(i))
        with open('{}recon_1x1_500.txt'.format(path)) as f:
             txt = f.readlines()
             gt_t = txt[0].split("|")
             r_t = txt[1].split("|")
             assemble_txtrecos(gt_t, r_t, '{}/recon_1x1_{}.png'.format(path, "eval_rec{}".format(i)))
        os.remove('{}recon_1x1_{}.txt'.format(path, "eval_rec{}".format(i)))
        if i == 5:
            break
    print("Saved visualizations for {}".format(path))
    return trainloader, testloader

def eval_reconstruct(path):
    model, args, trainloader, testloader = load_model(path, batch=1)
    device = torch.device("cuda")
    recons = []
    for i, data in enumerate(testloader):
        d = unpack_data(data[0], device=device)
        recon = model.reconstruct_data(d)
        recons.append(np.asarray(d[0].detach().cpu()))
        recons.append(np.asarray(recon[0].detach().cpu())[0])
        if i == 10:
            break
    with open(os.path.join(path, "visuals/reconstructions.pkl"), 'wb') as handle:
        pickle.dump(recons, handle)
    print("Saved reconstructions for {}".format(path))

def eval_analyse(path, K=1):
    model, args = load_model(path, batch=64)
    if model.modelName not in ["moe", "poe"]:
        enc = [model.enc_name]
    else:
        enc = model.encoders
    zss = []
    for i, e in enumerate(enc):
        l, data = load_data(e)
        d = [None] * len(enc)
        d[i] = data
        d = d[0] if len(enc) == 1 else d
        _, _, zs = model.forward(d, K=K)
        zs = zs[0] if isinstance(zs, list) else zs
        zss.append(zs.reshape(-1, zs.size(-1)).detach().cpu())
    t_sne(zss, path, "eval_mod{}".format(i), K, l)


def load_data(encoder):
    with open("/home/gabi/mirracle_remote/mirracle_multimodal/data/dataset2/labels.pkl", 'rb') as handle:
        labels = pickle.load(handle)
    if encoder == "Transformer":
       with open("/home/gabi/mirracle_remote/mirracle_multimodal/data/dataset2/action_data.pkl", 'rb') as handle:
            d = pickle.load(handle)
       d = [torch.from_numpy(np.asarray(x).astype(np.float)) for x in d]
       if len(d[0].shape) < 3:
           d = [torch.unsqueeze(i, dim=1) for i in d]
       d = torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0)
       d = [d[-128:], None]
    elif encoder == "Audio":
        with open("/home/gabi/mirracle_remote/mirracle_multimodal/data/dataset2/sounds.pkl", 'rb') as handle:
            d = pickle.load(handle)
        d = [torch.from_numpy(np.asarray(x).astype(np.int16)) for x in d]
        d = torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0).cuda()
    return labels[-128:], d[-128:]

def eval_sample(path):
    model, args, _, _ = load_model(path, batch=1)
    N, K = 36, 1
    samples = model.generate_samples(N, K).cpu().squeeze()
    with open(os.path.join(path, "visuals/latent_samples.pkl"), 'wb') as handle:
        pickle.dump(np.asarray(samples.detach().cpu()), handle)
    print("Saved samples for {}".format(path))

def plot_setup(xname, yname, pth, figname):
    plt.xlabel(xname)
    plt.ylabel(yname)
    #plt.legend()
    plt.savefig(os.path.join(pth, "visuals/{}.png".format(figname)))
    plt.clf()

def plot_loss(path):
    pth = os.path.join(path, "loss.csv") if not "loss.csv" in path else path
    loss = pd.read_csv(pth, delimiter=",")
    epochs = loss["Epoch"]
    losses = loss["Test Loss"]
    kld = loss["Test KLD"]
    l_mod = loss["Test Mod_0"]
    #plt.plot(epochs, l_mod, color='black', linestyle='dashed', label="Reconstruction Loss")
    plt.plot(epochs, losses, color='green', linestyle='solid', label="Loss")
    plot_setup("Epochs", "Loss", path, "loss")
    plt.plot(epochs, kld,  color='blue', linestyle='solid', label="KLD")
    plot_setup("Epochs", "KLD", path, "KLD")
    print("Saved loss plot")

def compare_models_numbers(pth):
    pth = pth + "/" if pth[-1] != "/" else pth
    f = open(os.path.join(pth, "comparison.csv"), 'w')
    writer = csv.writer(f)
    all_csvs = glob.glob(pth + "**/loss.csv", recursive=True)
    header = None
    for c in all_csvs:
        #plot_loss(c)
        model_csv = pd.read_csv(c, delimiter=",")
        if not header:
            writer.writerow(["Model", "Epochs"] + list(model_csv.keys())[1:])
            header = True
        row = [c.split(pth)[-1].split("/loss.csv")[0]] + [int(model_csv.values[-1][0])] + [round(x, 4) for x in list(model_csv.values[-1][1:])]
        writer.writerow(row)
    f.close()
    print("Saved comparison at {}".format(os.path.join(pth, "comparison.csv")))

def load_model(path, trainloader=None, testloader=None, batch=8):
    device = torch.device("cuda")
    config, mods, args = parse_args(cfg_path=path)
    model = "VAE" if len(mods) == 1 else config["mixing"].lower()
    modelC = getattr(models, model)
    params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods],
              [m["feature_dim"] for m in mods]]
    if len(mods) == 1:
        params = [x[0] for x in params]
    model = modelC(*params, config["n_latents"]).to(device)
    print('Loading model {} from {}'.format(model.modelName, path))
    model.load_state_dict(torch.load(os.path.join(path,'model.rar')))
    model._pz_params = model._pz_params
    model.eval()
    # if trainloader is None or testloader is None:
    #     trainloader, testloader = model.getDataLoaders(batch, device)
    return model, args


def print_txt2img(txt, correct = [1,1,1]):
    fnt = ImageFont.truetype(fonts[0], 14)
    image = Image.new(mode="RGB", size=(64, 64), color="white")
    draw = ImageDraw.Draw(image)
    for ix, word in enumerate(txt.split(" ")):
        color = (0,0,0) if correct[ix] == 1 else (256,0,0)
        draw.text((1, 5 + (ix*15)), word, font=fnt, fill=color)
    image = ImageOps.expand(image,border=1,fill='black')
    return np.asarray(image)

def cv_seeds(d):
    paths = glob(os.path.join(d,  "*/*/"))
    lis, lts = [], []
    for p in paths:
        if "{}d".format(VECDIM) in p:
                li, lt = read_loss(p)
                lis.append(li)
                lts.append(lt)
    print("MODEL: {}".format(p))
    print("Image: mean: {}  variance: {}   CV: {}%".format(statistics.mean(lis), statistics.stdev(lis),  round(100*statistics.stdev(lis)/statistics.mean(lis), 3)))
    print("Text: mean: {}  variance: {}  CV: {}%".format(statistics.mean(lts), statistics.stdev(lts), round(100*statistics.stdev(lts)/statistics.mean(lts),3)))


def assemble_txtrecos(gt, txt, pth):
    gts, rs = [], []
    o_l, r_l = [], []
    for i, x in enumerate(gt):
        gts.append(print_txt2img(x))
        c = []
        for ind, w in enumerate(x.split(" ")):
          if txt[i].split(" ")[ind].strip() == w.strip():
             c.append(1)
          else:
             c.append(0)
        rs.append(print_txt2img(txt[i], correct=c))
    for x in range(len(gt)):
        if o_l == []:
            o_l = np.asarray(gts[x])
            r_l = np.asarray(rs[x])
        else:
            o_l = np.hstack((o_l, np.asarray(gts[x])))
            r_l = np.hstack((r_l, np.asarray(rs[x])))
    w = np.vstack((o_l, r_l))
    w = cv2.copyMakeBorder(w,top=1, bottom=1, left=1, right=1,     borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    o_l = cv2.copyMakeBorder(o_l, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    r_l = cv2.copyMakeBorder(r_l, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
    r_l = cv2.cvtColor(r_l, cv2.COLOR_BGR2RGB)
    o_l = cv2.cvtColor(o_l, cv2.COLOR_BGR2RGB)
    cv2.imwrite(pth, w)
    cv2.imwrite(pth.replace(".png", "gt.png"), o_l)
    cv2.imwrite(pth.replace(".png", "recon.png"), r_l)

if __name__ == "__main__":
    p = "/home/gabi/mirracle_remote/mirracle_multimodal/mirracle_multimodal/results/actions_sounds_new"
    #plot_loss(p)
    eval_analyse(p)
    #compare_models_numbers(p)
    #eval_sample(p)
    #eval_reconstruct(p)

