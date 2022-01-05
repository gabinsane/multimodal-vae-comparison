import pkg_resources
import numpy as np
import os, pickle, random
from itertools import chain
from PIL import Image, ImageDraw, ImageFont
import math, glob, imageio, cv2
import argparse, sys
from train_w2v import train_word2vec
parser = argparse.ArgumentParser(description='VAE data generator')
parser.add_argument('--size', type=int, default=10000, help='size of the dataset')
parser.add_argument('--type', type=str, default="img-vec", help='type of the dataset. "img-img" =images + text images, "img-vec" = images + word embeddings ')
parser.add_argument('--emb-size', type=int, default=4096, help='Size of word vectors in the img-vec dataset')
parser.add_argument('--vecs-only', action='store_true', default=False, help='Generate additional embedding dataset based on attrs.pkl in data folder')
parser.add_argument('--noisytxt', action='store_true', default=False,
                    help='add noise to color names')
parser.add_argument('--subpart',  default=0,
                    help='0 keeps all categories balanced, 1-4 serve for incremental learning')
parser.add_argument('--noisycol', action='store_true', default=False,
                    help='add noise to image colors')
args = parser.parse_args()

noise = {"red":["reed", "red", "reen"], "green":["green", "greal", "greed"], "blue":["blue","bluack", "bleen"],
         "yellow":["yellow", "yealow","yeloon"], "purple":["purple", "blueprle", "purpleen"],
         "teal":["teal", "teavy", "tealoon"], "black":["black", "blavy", "bled"],
         "maroon":["maroon","mallow", "marple"], "navy":["navy", "nareen", "naveal"]}
fonts = ["/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf", "/usr/share/fonts/truetype/tlwg/Loma.ttf", "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
         "/usr/share/fonts/truetype/freefont/FreeSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"]
WORD_EMBEDDING_SIZE = args.emb_size
dimmap = {1:0, 4:1, 16:2, 64:3, 256:4, 1024:5, 4096:6}
sizes = ["small", "large"]
if int(args.subpart) == 0:
    shapes = ["line", "circle", "semicircle", "pieslice", "square"]
    colors = {"black": [0,0,0], "red": [255,0,0], "green": [0,255,0], "blue": [0,0,255], "grey": [128,128,128], "maroon": [105,0,0], "purple": [215,0,215], "teal": [0,175,175], "navy":[0,0,150], "orange":[255,140,0]}
elif int(args.subpart) == 1:
    shapes = ["line", "circle"]
    colors = {"black": [0,0,0], "red": [255,0,0], "green": [0,255,0]}
elif int(args.subpart) == 2:
    colors = {"orange": [0,0,255], "grey": [128,128,128], "maroon": [105,0,0]}
    shapes = ["semicircle", "pieslice"]
elif int(args.subpart) == 3:
    colors = {"purple": [215,0,215], "teal": [0,175,175], "navy":[0,0,150]}
    shapes = ["square",  "polygon"]


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
    print("\nMaking text images")
    os.makedirs(target_pth, exist_ok=True)
    target = unpickle(pth)
    for idx, word in enumerate(target):
        print("\r{}/{}".format(idx+1, len(target)), end = "")
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
        draw.text((random.uniform(3, 15), random.uniform(5, 35)), word[1].upper(), font=fnt, fill=(0, 0, 0))
        image.save(filename)

def word2vec(attrs_pth, target_pth):
    from gensim.models import Word2Vec
    words = unpickle(attrs_pth)
    if not os.path.exists("../data/word2vec{}d.model".format(WORD_EMBEDDING_SIZE)):
        print("Generating word2vec{}.model".format(WORD_EMBEDDING_SIZE))
        train_word2vec(words, vector_size=WORD_EMBEDDING_SIZE)
    model = Word2Vec.load("../data/word2vec{}d.model".format(WORD_EMBEDDING_SIZE))
    vecs = []
    print("Making {} (each word encoded into a {}d embedding)".format(target_pth, WORD_EMBEDDING_SIZE))
    for seq in words:
        vecs.append(([model.wv[seq[0]]],[model.wv[seq[1]]], [model.wv[seq[2]]]))
    vecs_np = np.asarray(vecs).squeeze()
    with open(target_pth, 'wb') as handle:
        pickle.dump(vecs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    print("Making ../data/attrs.pkl")
    attrs = []
    for x in range(args.size):
        size, color, shape = "", "", ""
        if sizes:
            size = random.choice(sizes)
        if colors:
            color = random.choice(list(colors.keys()))
        if shapes:
            shape = random.choice(shapes)
        attrs.append([size, color,shape])
    attrs = np.asarray(attrs)
    with open(os.path.join(path, './attrs.pkl'), 'wb') as handle:
        pickle.dump(attrs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle(pth):
    with open(pth, 'rb') as handle:
        target = pickle.load(handle)
        #target = np.expand_dims(text, axis=1).tolist()  # only takes first word from the sequences
        #target = list(chain.from_iterable(text))
    return target

def make_shape_imgs(pth, target_pth):
    print("Making {}".format(target_pth))
    target = unpickle(pth)
    for idx, text in enumerate(target):
        size = text[0]
        colname = text[1]
        shapename = text[2]
        print("\r{}/{}".format(idx+1, len(target)), end = "")
        # name of the file to save
        padding = 6 - len(str(idx))
        index = padding * "0" + str(idx)
        filename = os.path.join(target_pth, "img_{}.png".format(index))
        image = Image.new(mode="RGB", size=(64, 64), color="white")
        draw = ImageDraw.Draw(image)
        color = randomize_rgb(colors[colname]) if args.noisycol else colors[colname]
        size_add = 30 if size == "large" else 16
        x1 = random.randint(5,35)
        x2 = random.randint(5,35)
        shapes = {"circle": draw.ellipse, "line":draw.line, "square":draw.rectangle, "semicircle":draw.chord, "pieslice":draw.pieslice,  "polygon":draw.polygon}
        shape = shapes[shapename]
        if shape not in [draw.chord, draw.pieslice, draw.polygon]:
            shape((x1, x2, x1+size_add, x2+size_add), fill=tuple(color), width=int(size_add/2))
        else:
            if shape == draw.polygon:
                shape(((x1, x2), (x2, x1), (x2 + size_add, x1 + size_add)), fill=tuple(color), outline=(0, 0, 0))
            else:
                coords = [50,270] if shape == draw.chord else [200,250]
                size_a = size_add if shape == draw.chord else size_add *2
                shape((x1, x2, x1 + size_a, x2+size_a), start=coords[0], end=coords[1], fill=tuple(color))
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

def make_dummy_imgs(pth, target_pth):
    print("Making color images")
    target = unpickle(pth)
    for idx, word in enumerate(target):
        print("\r{}/{}".format(idx+1, len(target)), end = "")
        # name of the file to save
        padding = 6 - len(str(idx))
        index = padding * "0" + str(idx)
        filename = os.path.join(target_pth, "img_{}.png".format(index))
        image = Image.new(mode="RGB", size=(64, 64), color="white")
        draw = ImageDraw.Draw(image)
        color = randomize_rgb(colors[word[1]]) if args.noisycol else colors[word[1]]
        x1 = random.randint(0,30)
        x2 = random.randint(0,30)
        #  draw.ellipse((x1, x2, x1 + 30, x2+30), fill=word, outline=word)
        draw.line((x1, x2, x1 + 30, x2+30), fill=tuple(color), width=10)
        image.save(filename)

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
    os.makedirs("../../data", exist_ok=True)
    target_dir = "../../data"
    if not args.vecs_only:
        make_attrs(target_dir)
        os.makedirs(os.path.join(target_dir, "image"), exist_ok=True)
    else:
        print("Using the pre-generated attrs.pkl to make {}d.pkl".format(args.emb_size))
    if args.type == "img-img":
       make_dummy_imgs(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "image"))
       os.makedirs(os.path.join(target_dir, "imagetxt"), exist_ok=True)
       make_dummy_txt(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "imagetxt"))
    elif args.type == "img-vec":
       word2vec(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "word2vec{}d.pkl".format(WORD_EMBEDDING_SIZE)))
       if not args.vecs_only:
        make_shape_imgs(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "image"))
    print("\nAll done. Data saved in mirracle_multimodal/data")