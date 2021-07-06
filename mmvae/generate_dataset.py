import pkg_resources
import numpy as np
import os, pickle, random
from itertools import chain
from PIL import Image, ImageDraw, ImageFont
import math, glob, imageio, cv2
import argparse
parser = argparse.ArgumentParser(description='VAE data generator')
parser.add_argument('--size', type=int, default=5000, help='size of the dataset')
parser.add_argument('--noisytxt', action='store_true', default=False,
                    help='add noise to color names')
parser.add_argument('--noisycol', action='store_true', default=False,
                    help='add noise to image colors')
args = parser.parse_args()

noise = {"red":["reed", "red", "reen"], "green":["green", "greal", "greed"], "blue":["blue","bluack", "bleen"],
         "yellow":["yellow", "yealow","yeloon"], "purple":["purple", "blueprle", "purpleen"],
         "teal":["teal", "teavy", "tealoon"], "black":["black", "blavy", "bled"],
         "maroon":["maroon","mallow", "marple"], "navy":["navy", "nareen", "naveal"]}
fonts = ["/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf", "/usr/share/fonts/truetype/tlwg/Loma.ttf", "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
         "/usr/share/fonts/truetype/freefont/FreeSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"]
dimmap = {1:0, 4:1, 16:2, 64:3, 256:4, 1024:5, 4096:6}
colors = {"black": [0,0,0], "red": [255,0,0], "green": [0,255,0], "blue": [0,0,255], "yellow": [255,255,0], "maroon": [105,0,0], "purple": [215,0,215], "teal": [0,175,175], "navy":[0,0,150]}

def make_text_img(datapath, txt, idx):
    print("Image {}".format(idx))
    # name of the file to save
    padding = 6 - len(str(idx))
    index = padding * "0" + str(idx)
    filename = os.path.join(datapath, "img_{}.png".format(index))
    fnt = ImageFont.truetype(fonts[3], 15)
    image = Image.new(mode="RGB", size=(64, 64), color="white")
    draw = ImageDraw.Draw(image)
    text = random.choice(noise[txt]) if args.noisytxt else txt
    draw.text((random.uniform(3,15), random.uniform(5,35)), text, font=fnt, fill=(0, 0, 0))
    image.save(filename)

def make_dummy_txt(pth, target_pth):
    print("making text images")
    with open(pth, 'rb') as handle:
        text = pickle.load(handle)
        target = np.expand_dims(text[:,0], axis=1).tolist()  # only takes first word from the sequences
        target = list(chain.from_iterable(target))
    for idx, word in enumerate(target):
        print("Image {}".format(idx))
        # name of the file to save
        padding = 6 - len(str(idx))
        index = padding * "0" + str(idx)
        filename = os.path.join(target_pth, "img_{}.png".format(index))
        image = Image.new(mode="RGB", size=(64, 64), color="white")
        draw = ImageDraw.Draw(image)
        try:
            fnt = ImageFont.truetype(fonts[3], 13)
        except:
            fnt = ImageFont.load_default()
        draw.text((random.uniform(3, 15), random.uniform(5, 35)), word.upper(), font=fnt, fill=(0, 0, 0))
        image.save(filename)

def randomize_rgb(rgb):
    new_rgb = [0,0,0]
    for ix, c in enumerate(rgb):
         if c == 255:
             new_rgb[ix] = c - random.randint(0,70)
         elif c == 0:
             new_rgb[ix] = c + random.randint(0,70)
         else:
             new_rgb[ix] = c + random.randrange(-40, 40)
    return new_rgb

def make_attrs(path):
    print("making attrs.pkl")
    attrs = []
    for x in range(args.size):
        attrs.append([random.choice(list(colors.keys()))])
    attrs = np.asarray(attrs)
    with open(os.path.join(path, './attrs.pkl'), 'wb') as handle:
        pickle.dump(attrs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_dummy_imgs(pth, target_pth):
    print("making color images")
    with open(pth, 'rb') as handle:
        text = pickle.load(handle)
        target = np.expand_dims(text[:,0], axis=1).tolist()  # only takes first word from the sequences
        target = list(chain.from_iterable(target))
    for idx, word in enumerate(target):
        print("Image {}".format(idx))
        # name of the file to save
        padding = 6 - len(str(idx))
        index = padding * "0" + str(idx)
        filename = os.path.join(target_pth, "img_{}.png".format(index))
        image = Image.new(mode="RGB", size=(64, 64), color="white")
        draw = ImageDraw.Draw(image)
        color = randomize_rgb(colors[word]) if args.noisycol else colors[word]
        x1 = random.randint(0,30)
        x2 = random.randint(0,30)
        #  draw.ellipse((x1, x2, x1 + 30, x2+30), fill=word, outline=word)
        draw.line((x1, x2, x1 + 30, x2+30), fill=tuple(color), width=10)
        image.save(filename)

def make_arrays(dataset, pth, name):
    d = []
    dims = int(name.split("D")[0])
    c = 1
    for img in dataset:
        print("{}/{}".format(c,dataset.shape[0]))
        c += 1
        subims = [img]
        for _ in range(dimmap[dims]):
                inter_l = []
                for i in subims:
                    i_part = int(i.shape[0] / 2)
                    inter_l.extend([i[:i_part, :i_part], i[i_part:, :i_part], i[:i_part, i_part:], i[i_part:, i_part:]])
                subims = inter_l
        i = [np.round(x.mean(2).mean(0).mean(0), 5) for x in subims]
        d.append(i)
    d = np.asarray(d)
    with open(os.path.join(pth, '{}.pkl'.format(name)), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("SAVED {}".format(os.path.join(pth, '{}.pkl'.format(name))))


def load_images(path, imsize=64, size=math.inf):
    print("Loading data...")
    images = sorted(glob.glob(os.path.join(path, "*.png")))
    dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
    for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            # image = reshape_image(image, self.imsize)
            image = cv2.resize(image, (imsize, imsize))
            if i >= size:
                break
            dataset[i, :] = image /255
    print("Dataset of shape {} loaded".format(dataset.shape))
    return dataset

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    target_dir = "../data"
    make_attrs(target_dir)
    os.makedirs(os.path.join(target_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "imagetxt"), exist_ok=True)
    make_dummy_imgs(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "image"))
    make_dummy_txt(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "imagetxt"))
    print("All done. Data saved in mirracle_multimodal/data")
