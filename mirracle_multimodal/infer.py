import torch, numpy as np
from mirracle_multimodal import models
from mirracle_multimodal.utils import unpack_data
from PIL import Image, ImageDraw, ImageFont,  ImageOps
import cv2, os
from glob import glob

fonts = ["/usr/share/fonts/truetype/freefont/FreeSans.ttf"]

def eval(path, trainloader=None, testloader=None):
    args = torch.load(os.path.join(path, 'args.rar'))
    args.mod1 = "/home/gabi/mirracle_multimodal/data/image"
    args.mod2 = "/home/gabi/mirracle_multimodal/data/3d.pkl"
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
        trainloader, testloader = model.getDataLoaders(16, False, device)
    for i, data in enumerate(testloader):
        if "2mods" in args.model:
            d = unpack_data(data, device=device)
        else:
            d = unpack_data(data[0], device=device)
        model.reconstruct(d, path, "eval_rec{}".format(i))
        with open('{}recon_1x1_{}.txt'.format(path, "eval_rec{}".format(i))) as f:
            txt = f.readlines()
            gt_t = txt[0].split("|")
            r_t = txt[1].split("|")
            assemble_txtrecos(gt_t, r_t, '{}/recon_1x1_{}.png'.format(path, "eval_rec{}".format(i)))
        os.remove('{}recon_1x1_{}.txt'.format(path, "eval_rec{}".format(i)))
        if i == 15:
            break
    print("Saved visualizations for {}".format(path))
    return trainloader, testloader

def print_txt2img(txt):
    fnt = ImageFont.truetype(fonts[0], 14)
    image = Image.new(mode="RGB", size=(64, 64), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((1, 5), txt.replace(" ", "\n"), font=fnt, fill=(0, 0, 0))
    image = ImageOps.expand(image,border=1,fill='black')
    return np.asarray(image)

def assemble_txtrecos(gt, txt, pth):
    gts, rs = [], []
    o_l, r_l = [], []
    for i, x in enumerate(gt):
        gts.append(print_txt2img(x))
        rs.append(print_txt2img(txt[i]))
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
    cv2.imwrite(pth, w)
    cv2.imwrite(pth.replace(".png", "gt.png"), o_l)
    cv2.imwrite(pth.replace(".png", "recon.png"), r_l)

trainloader, testloader = None, None
paths = "/home/gabi/mirracle_multimodal/mmvae/results/sm/8L/bce/3d/3d_8l_moe_bce/1d/"
dir_of_models = "/home/gabi/mirracle_multimodal/mmvae/results/sm"
if dir_of_models:
    paths = glob(os.path.join(dir_of_models,  "*/*/*/*/*/"))
    for p in paths:
        if "3d" in p:
                trainloader, testloader = eval(p)
else:
    eval(paths)


