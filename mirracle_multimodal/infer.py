import torch, numpy as np
import models
from utils import unpack_data
from PIL import Image, ImageDraw, ImageFont,  ImageOps
import cv2, os
from glob import glob
import pandas as pd
import math, statistics
import matplotlib.pyplot as plt

fonts = ["/usr/share/fonts/truetype/freefont/FreeSans.ttf"]
VECDIM = 4096
plot_colors = ["blue", "green", "red", "cyan", "magenta", "orange", "navy", "maroon", "brown"]


def read_loss(path):
    loss = pd.read_csv(os.path.join(os.path.dirname(path), "loss.csv"), delimiter=",")
    print(path)
    if "Test Image Loss" in loss.keys():
        l1i = loss["Test Image Loss"][499]
        print("IMG LOSS 500: {}".format(round(l1i, 2)))
    if "Test ImageTxt Loss" in loss.keys():
        l1t = loss["Test ImageTxt Loss"][499]
        #print("TXT LOSS 500: {}".format(round(l1t, 2)))
    return l1i, l1t

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

def eval1(path, trainloader=None, testloader=None):
    model, args, trainloader, testloader = load_model(path, batch=32)
    device = torch.device("cuda" if args.cuda else "cpu")
    for i, data in enumerate(testloader):
        if "2mods" in args.model:
            d = unpack_data(data, device=device)
        else:
            d = unpack_data(data[0], device=device)
        model.analyse_encodings(d, path, 1)
        if i == 0:
            break
    print("Saved visualizations for {}".format(path))
    return trainloader, testloader

def load_model(path, trainloader=None, testloader=None, batch=8):
    args = torch.load(os.path.join(path, 'args.rar'))
    args.mod1 = "/home/gabi/mirracle_multimodal/data/image"
    if "4096d_" in path:
        vecdim = 4096
    else:
        vecdim = 3
    args.mod2 = "/home/gabi/mirracle_multimodal/data/{}d.pkl".format(vecdim)
    device = torch.device("cuda" if args.cuda else "cpu")
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    if args.model == "uni":
        model = modelC(args, index=0).to(device)
    else:
        model = modelC(args).to(device)
    print('Loading model {} from {}'.format(model.modelName, path))
    model.load_state_dict(torch.load(os.path.join(path,'model.rar')))
    model._pz_params = model._pz_params
    model.eval()
    if trainloader is None or testloader is None:
        trainloader, testloader = model.getDataLoaders(batch, False, device)
    return model, args, trainloader, testloader


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
                li, lt = read_loss(p) #trainloader, testloader = eval(p)
                lis.append(li)
                lts.append(lt)
    print("MODEL: {}".format(p))
    print("Image: mean: {}  variance: {}   CV: {}%".format(statistics.mean(lis), statistics.stdev(lis),  round(100*statistics.stdev(lis)/statistics.mean(lis), 3)))
    print("Text: mean: {}  variance: {}  CV: {}%".format(statistics.mean(lts), statistics.stdev(lts), round(100*statistics.stdev(lts)/statistics.mean(lts),3)))

def plot_results(compare, fig, log, ix, first=True):
    rows = 2
    cols = 2
    ax = fig.add_subplot(rows, cols, ix)
    for ix, pth in enumerate(sorted(compare)):
        seed = pth.split("/")[-2][-1] if pth.split("/")[-2][-1] not in ["l", "e"] else "1"
        data = pd.read_csv(os.path.join(pth, "loss.csv"), delimiter=",")
        name = pth.split("/")[-4].upper().replace("O", "o")
        label = "{} SEED {}".format(name, seed)
        try:
            ax.plot(data[log], label=label, color=plot_colors[ix])
            ax.set_xlim(xmin=0, xmax=500)
            ylim = [500, 2500] if pth.split("/")[-3] == "bce" else [7750,8250]
            ax.set_ylim(ylim)
            ax.grid()
            ax.legend(loc="upper right", title="Model")
        except:
            print("Not plotting {} for {}".format(log, pth))
            pass
    ax.set_xlabel("Epoch")
    ax.set_ylabel(pth.split("/")[-3].upper(), fontweight='bold')
    ax.set_title("SCS-IMG {}".format(name))
    return fig

def plot_convergence(paths):
    log = ["Test Image Loss"]
    nll = [p for p in paths if p.split("/")[-3] == "nll"]
    bce = [p for p in paths if p.split("/")[-3] == "bce"]
    bce_poe = [p for p in bce if p.split("/")[-4] == "poe"]
    bce_moe = [p for p in bce if p.split("/")[-4] == "moe"]
    nll_poe = [p for p in nll if p.split("/")[-4] == "poe"]
    nll_moe = [p for p in nll if p.split("/")[-4] == "moe"]
    group = [bce_moe, bce_poe, nll_moe, nll_poe]
    fig = plt.figure()
    #fig.subplots_adjust(hspace=0.5, top=0.8)
    first = True
    for ix, l in enumerate(group):
        fig = plot_results(l, fig, log, ix + 1, first)
        first = False
    fig.tight_layout()
    plt.show()


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

trainloader, testloader = None, None
paths = "/home/gabi/mirracle_multimodal/mmvae/results/single/img_24l_bce/0a/"
dir_of_models = "/home/gabi/mirracle_remote/mirracle_multimodal/mirracle_multimodal/results/BAL/24L"
#cv_seeds(dir_of_models)
if dir_of_models:
    paths = glob(os.path.join(dir_of_models, "*/*/*/*"))
    plot_convergence(paths)
    # for p in paths:
    #     #if not "txt" in p and not 'img' in p:
    #     try:
    #         eval(p)
    #     except:
    #         pass
else:
    eval1(paths)


