"""
Code for generation of the CdSprites+ dataset.

This code is adapted from https://github.com/yordanh/interp_latent_spaces featuring dSprites with color, published as a paper
"Interpretable Latent Spaces for Learning from Demonstration".
We changed some functions, added textures to the shapes and backgrounds and also added natural language captions.
We also created 5 difficulty levels.

Original code: https://github.com/yordanh/interp_latent_spaces/blob/master/src/preprocess_dsprites.py
author          :Yordan Hristov <yordan.hristov@ed.ac.uk>
date            :05/2018
python_version  :2.7.14

Modification:
author          :Gabriela Sejnova <gabriela.sejnova@cvut.cz>
date            :05/2023
python_version  :3.8
==============================================================================
"""

import numpy as np
import cv2
import numpy, random
import argparse
import os, glob
import shutil
import itertools
import copy
import json
import wget
import tarfile
import h5py

dsprites_path = "./dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
if not os.path.exists(dsprites_path):
    url = 'https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    wget.download(url)
parser = argparse.ArgumentParser(description='Process the dSprited dataset.')
parser.add_argument('--image_size', default=64, type=int, help='Width and height of the images in px.')
parser.add_argument('--level', default=0, type=int, help='If you only want to generate one level of the dataset')



class ConfigParser(object):
    def __init__(self, filename):
        file = open(filename, "r")
        self.config = json.load(file)

    def parse_specs(self):
        specs = {'train': [], "unseen":[]}
        specs['train'] = self.config['data_generation']['train']['spec']
        # for record in self.config['data_generation']['unseen']:
        #     specs['unseen'].append((record['label'], record['spec']))
        return specs

def change_brightness(img, value=30, increase=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if increase:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = 0 + value
        v[v < lim] = 0
        v[v <= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def colorize_background(image, texture):
    ### Replaces black pixels in the image with the background texture
    tex = cv2.resize(texture, image.shape[:2])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray_image == 0
    out = cv2.bitwise_and(tex, tex, image, mask.astype(np.uint8))
    outp = cv2.normalize(out, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return outp

def colorize_texture(texture, rgb):
    ### Takes the loaded texture image (RGB, 0-255) and changes color based on the RGB array (0-255)
    texture = texture_to_bnw(texture, shade="light")
    return texture * rgb/255

def texture_to_bnw(texture, shade=None):
    tex = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    tex = numpy.tile(tex.reshape(tex.shape[0], tex.shape[1], 1), (1, 1, 3))
    outp = cv2.normalize(tex, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if shade is not None:
        if shade == "light":
            outp = change_brightness(outp, 125)
        else:
            outp = change_brightness(outp, 200, increase=False)
    return outp

def make_textured_shape(image, texture):
    ### Takes the input RGB image with black background and adds the colored texture to the shape
    out = image * cv2.resize(texture, image.shape[:2])
    outp = cv2.normalize(out, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return outp

def prep_dir(folder_name):
    print("Preparing " + folder_name)
    if os.path.exists(folder_name):
        print("Cleaning " + folder_name)
        map(lambda object_folder: shutil.rmtree(folder_name + object_folder), os.listdir(folder_name))
        print(folder_name + " has been cleaned!")
    else:
        os.makedirs(folder_name)


# extracts images for a single label - e.g big and blue in the label group big_blue
def extract(folder_name=None, labels=None, args=None, latent_spec=None, image_size=None, verbose=False):
    print("Extracting images for " + folder_name + str(labels))
    if "position" not in latent_spec.keys():
        x_pose = latent_spec['x']
        y_pose = latent_spec['y']
    else:
        x_pose = latent_spec['position'][0]
        y_pose = latent_spec['position'][1]
    indices = []
    for i, c in enumerate(data['latents_classes']):
        if (c[1] in latent_spec['shape'] and
                c[2] in latent_spec['scale'] and
                c[3] in latent_spec['orientation'] and
                c[4] in x_pose and
                c[5] in y_pose):
            indices.append(i)

    random.shuffle(indices)
    images = numpy.take(data['imgs'], indices[:sample_num], axis=0)

    for i, image in enumerate(images):
        image = cv2.resize(image, (image_size, image_size))

        for bgr_color in latent_spec["color"]:
            if len(latent_spec["textured"]) == 0:
                image_out = numpy.tile(image.reshape(image_size, image_size, 1), (1, 1, 3)) * bgr_color
                image_out = cv2.cvtColor(image_out.astype(np.uint8), cv2.COLOR_BGR2RGB)
            else:
                if "shapes" in latent_spec["textured"]:
                    t = random.randint(0, len(textures)-1)
                    image_out = make_textured_shape(numpy.tile(image.reshape(image_size, image_size, 1), (1, 1, 3)), colorize_texture(textures[t], bgr_color))
                if "background" in latent_spec["textured"]:
                    image_out = colorize_background(image_out, texture_to_bnw(random.choice(textures), shade=labels[-1].split("_")[-1]))
                image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
            object_folder_name = folder_name + "_".join(labels) + "/"
            cv2.imwrite(object_folder_name + "/" + str(label_counters["_".join(labels)]) + ".png", image_out)
            label_counters["_".join(labels)] += 1

        if verbose:
            cv2.imshow("image", image_out)
            cv2.waitKey()

        if i % 100 == 0:
            print("{0} images have been processed so far.".format(i))
    print(label_counters)


# extracts symbols for a label group - e.g big_blue
def extract_label_groups(label_groups=None, unseen=None, latent_spec=None, mappings=None, folder_name=None, args=None):
    # build up the labels for all objects - take the combinations of the
    # lists in label_groups
    textured = label_groups["textured"]
    label_groups.pop("textured")
    if len(textured) > 0 and not "level3" in cfg:
        label_groups["position"] = ["at_top_right", "at_top_left", "at_bottom_left", "at_bottom_right"]
    if "background" in textured:
        label_groups["textured"] = ["on_light", "on_dark"]
    object_labels = list(itertools.product(*[label_groups[x] for x in label_groups]))
    for labels in object_labels:
        object_folder_name = folder_name + "_".join(labels) + "/"
        os.makedirs(object_folder_name, exist_ok=True)
        revised_latent_spec = revise_latent_spec(copy.deepcopy(latent_spec), labels, mappings)
        label_counters["_".join(labels)] = 0
        revised_latent_spec["textured"] = textured
        extract(folder_name=folder_name, labels=labels, args=args, latent_spec=revised_latent_spec,
                image_size=args.image_size)


# revise the latent class specification, depending on the
# given labels; we know what labels map to what classes
# across the different factors of variation
def revise_latent_spec(latent_spec, label, mappings):
    colors = latent_spec["color"]
    latent_spec["color"] = []
    for color in colors:
        latent_spec["color"] += mappings["color"][color]
    mappings_keys = mappings.keys()
    for key in label:
        for mkey in mappings_keys:
            if key in mappings[mkey].keys():
                new_value = mappings[mkey][key]
                if isinstance(new_value, list):
                    latent_spec[mkey] = new_value
                else:
                    latent_spec[mkey] = [new_value]
                break
    return latent_spec

def download_textures(dtd_path):
    print("Downloading textures...")
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
    wget.download(url)
    file = tarfile.open(dtd_path)
    file.extractall('./dtd_textures')
    file.close()

def load_textures(dtd_path="./dtd_textures"):
    if not os.path.exists(dtd_path):
        download_textures(dtd_path)
    textures_paths = glob.glob(os.path.join(dtd_path, './*/*/*/*.jpg'))
    textures = []
    for i in textures_paths:
        textures.append(cv2.imread(i))
    return textures

if __name__ == "__main__":
    args = parser.parse_args()
    data = np.load(dsprites_path)
    label_counters = {}
    #textures = load_textures()
    if args.level == 0:
        cfgs = ["config_level2.json", "config_level3.json", "config_level4.json", "config_level5.json"]
    else:
        cfgs = ["config_level{}.json".format(args.level)]
    for cfg in cfgs:
        config_parser = ConfigParser(cfg)
        sample_num = config_parser.config["samples_num"]
        mappings = {}
        mappings['color'] = {'white': [numpy.array([255, 255, 255])],
                             'red': [numpy.array([192, 64, 0])],
                             'yellow': [numpy.array([228, 217, 111])],
                             'green': [numpy.array([10, 107, 60])],
                             'blue': [numpy.array([0, 127, 200])],
                             'pink': [numpy.array([255, 0, 255])]
                             }
        mappings['shape'] = {'square': 0, 'ellipse': 1, 'heart': 2}
        mappings['scale'] = {'small': [0], 'medium': [2], 'big': [5]}
        mappings['orientation'] = {'rotated': [4, 14, 24, 34], 'flat': [0, 10, 20, 39]}
        mappings["position"] = {"at_top_right":[[26,27,28,29,30,31],[1,2,3,5,6,7,8,9]],
                                "at_top_left":[[1,2,3,5,6,7,8,9],[1,2,3,5,6,7,8,9]],
                                "at_bottom_left":[[1,2,3,5,6,7,8,9],[26,27,28,29,30,31]],
                                "at_bottom_right":[[26,27,28,29,30,31],[26,27,28,29,30,31]]}

        # describes the specification wrt to which we filter the
        # images, depending on their latent factor classes
        # the spec is refined once we are given labels
        latent_spec = {'color': ['white', 'red', 'yellow', 'green', 'blue', 'pink'],
                       'shape': [0,1,2],  # range(3),
                       'scale': [0, 5],  # range(6),
                       'orientation': range(45),
                       'x': [5,6,7,8,9,10,11,12,13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29],
                       'y': [5,6,7,8,9,10,11,12,13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29],}

        # delete any previous object folders
        folder_name = "../data/CdSpritesplus/{}/".format(cfg.split("_")[-1].split(".json")[0])
        prep_dir(folder_name)
        specs = config_parser.parse_specs()

        # extract_label_groups(label_groups=specs["train"], folder_name=folder_name + "train/", latent_spec=latent_spec,
        #                      mappings=mappings, args=args)
        images = glob.glob(os.path.join(folder_name, '*/*/*.png'))
        imgs = []
        captions = []
        for ind, i in enumerate(images):
            print("Compressing dataset image {}/{}".format(ind, len(images)))
            im = cv2.imread(i)
            imgs.append(im)
            caption = os.path.basename(os.path.dirname(i)).replace("_", " ")
            if "level1" in cfg:
                caption = caption.split(" ")[-1]
            elif "level2" in cfg:
                caption = " ".join([caption.split(" ")[0],caption.split(" ")[-1]])
            captions.append(caption)
        hf = h5py.File(os.path.join(folder_name, 'traindata.h5'), 'w')
        hf.create_dataset('image', data=np.asarray(imgs))
        hf.create_dataset('text', data=captions)
        hf.close()


