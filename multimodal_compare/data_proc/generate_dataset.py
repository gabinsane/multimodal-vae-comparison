import numpy as np
import os, pickle, random
from PIL import Image, ImageDraw
import argparse
from math import cos, sin, pi

parser = argparse.ArgumentParser(description='GeBiD data generator')
parser.add_argument('--dir', type=str, default="../data/level55", help='where to save the dataset the dataset')
parser.add_argument('--level', type=int, default=5, help='difficulty level: 1-5')
parser.add_argument('--size', type=int, default=100, help='size of the dataset')
parser.add_argument('--noisycol', action='store_true', default=False,
                    help='add noise to image colors')
args = parser.parse_args()

shapes = ["line", "circle", "square", "semicircle", "pieslice", "spiral"]
colors = {"yellow": [255, 255, 0], "red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255],
          "grey": [128, 128, 128], "brown": [105, 0, 0], "purple": [215, 0, 215], "teal": [0, 175, 175],
          "navy": [0, 0, 150], "orange": [255, 140, 0], "beige": [232, 211, 185], "pink": [255, 182, 193]}
sizes = ["small", "large"]
locations1 = ["at the top", "at the bottom"]
locations2 = ["left", "right"]
backgrounds = ["on white", "on black"]

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

def draw_spiral(x_l, y_l, a, b, img, colour, step=0.5, loops=5):
    """
    Draw the Archimdean spiral defined by:
    r = a + b*theta
    Args:
        a (real): First parameter
        b (real): Second parameter
        img (Image): Image to write spiral to.
        step (real): How much theta should increment by. (default: 0.5)
        loops (int): How many times theta should loop around. (default: 5)
    """
    draw = ImageDraw.Draw(img)
    theta = 0.0
    r = a
    prev_x = int(r*cos(theta))
    prev_y = int(r*sin(theta))
    while theta < 2 * loops * pi:
        theta += step
        r = a + b*theta
        # Draw pixels, but remember to convert to Cartesian:
        x = int(r*cos(theta))
        y = int(r*sin(theta))
        draw.line((prev_x+x_l, prev_y+y_l) + ((x+x_l),(y+y_l)), fill=tuple(colour))
        prev_x = x
        prev_y = y

def make_attrs(path):
    print("Making ../data/attrs.pkl")
    attrs = []
    for x in range(args.size):
        size, color, shape, loc1, loc2, bkgr = "", "", "", "", "", ""
        if sizes:
            size = random.choice(sizes)
        if colors:
            color = random.choice(list(colors.keys()))
        if shapes:
            shape = random.choice(shapes)
        loc1 = random.choice(locations1)
        loc2 = random.choice(locations2)
        bkgr = random.choice(backgrounds)
        attrs.append([size, color, shape, loc1, loc2, bkgr])
    attrs = np.asarray(attrs)
    with open(os.path.join(path, './attrs.pkl'), 'wb') as handle:
        pickle.dump(attrs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle(pth):
    with open(pth, 'rb') as handle:
        target = pickle.load(handle)
    return target

def pickle_dump(pth, data):
    with open(pth, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def text_to_level(text, level):
    filters = [None, [t[2] for t in text], [list(list([t[0]]) + list([t[2]])) for t in text],
               [list(t[:3]) for t in text], [(list(t[:3]) + [" ".join(t[-1:])]) for t in text], [(list(t[:3]) + [" ".join(t[3:5])] + [t[-1]]) for t in text]]
    text = filters[level]
    return text

def make_shape_imgs(pth, target_pth, imglevel=5, txtlevel=5):
    print("Making {}".format(target_pth))
    target = unpickle(pth)
    target_modified = text_to_level(target, txtlevel)
    pickle_dump(pth, target_modified)
    for idx, text in enumerate(target):
        size = text[0]
        colname = text[1]
        shapename = text[2]
        print("\r{}/{}".format(idx+1, len(target)), end = "")
        # name of the file to save
        padding = 6 - len(str(idx))
        index = padding * "0" + str(idx)
        filename = os.path.join(target_pth, "img_{}.png".format(index))
        bkgr = text[5].split(" ")[-1] if imglevel >= 4 else "white"
        image = Image.new(mode="RGB", size=(64, 64), color=bkgr)
        draw = ImageDraw.Draw(image)
        color = randomize_rgb(colors[colname]) if args.noisycol else colors[colname]
        if imglevel < 3:
            color = [0,0,0]
        if imglevel > 1:
            size_add = 30 if size == "large" else 16
        else:
            size_add = 30
        if imglevel == 5:
            x1 = random.randint(5,10) if "left" in text[4] else random.randint(30,35)
            x2 = random.randint(5,10) if "top" in text[3] else random.randint(30,35)
        else:
            x1, x2 = 20 + random.randint(-1,1), 20 + random.randint(-1,1)
        shapes = {"circle": draw.ellipse, "line":draw.line, "square":draw.rectangle, "semicircle":draw.chord, "pieslice":draw.pieslice,  "polygon":draw.polygon, "spiral":draw_spiral}
        shape = shapes[shapename]
        if shape not in [draw.chord, draw.pieslice, draw.polygon, draw_spiral]:
            shape((x1, x2, x1+size_add, x2+size_add), fill=tuple(color), width=int(size_add/2))
        else:
            if shape == draw.polygon:
                shape(((x1, x2), (x2, x1), (x2 + size_add, x1 + size_add)), fill=tuple(color), outline=(0, 0, 0))
            elif shape in [draw.pieslice, draw.chord]:
                coords = [50,270] if shape == draw.chord else [200,250]
                size_a = size_add if shape == draw.chord else size_add *2
                shape((x1, x2, x1 + size_a, x2+size_a), start=coords[0], end=coords[1], fill=tuple(color))
            else:
                if imglevel > 1:
                    s = 0.6 if size == "large" else 0.3
                else:
                    s = 0.6
                if imglevel < 5:
                    x1,x2 = 32, 32
                shape(x1, x2, s, s, image, color)
        image.save(filename)

if __name__ == "__main__":
    target_dir = args.dir
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "image"), exist_ok=True)
    make_attrs(target_dir)
    make_shape_imgs(os.path.join(target_dir, "attrs.pkl"), os.path.join(target_dir, "image"), imglevel=args.level, txtlevel=args.level)
    print("\nAll done. Data saved in {}".format(target_dir))