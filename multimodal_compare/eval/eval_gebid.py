### code for qualitative evaluation of multimodal VAEs trained on GeBiD dataset
from eval.infer import load_model, text_to_image, get_traversal_samples
import cv2, pickle
from math import sqrt
import os, glob, imageio
from main import Trainer
import numpy as np
import torch
from PIL import Image
from utils import output_onehot2text
import glob
import argparse

shapetemplates = glob.glob('./eval/templates/*.png')
colors = {"yellow": [255, 255, 0], "red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255],
          "grey": [128, 128, 128], "brown": [105, 0, 0], "purple": [215, 0, 215], "teal": [0, 175, 175],
          "navy": [0, 0, 150], "orange": [255, 140, 0], "beige": [232, 211, 185], "pink": [255, 182, 193],
          "black":[0,0,0], "white":[255,255,255]}
shapenames = ["spiral", "line", "square", "semicircle","circle", "pieslice"]
sizes = ["small", "large"]
locations1 = ["at the top left", "at the top right", "at the bottom right",  "at the bottom left"]
locations2 = ["left", "right"]
backgrounds = ["on white", "on black"]
level_attributes = {1:["shape"],2:["size","shape"],3:["size", "color","shape"],
                    4:["size", "color", "shape", "background"],5:[ "size", "color", "shape", "position","background"]}
sources = {"shape": shapenames, "size": sizes, "color": list(colors.keys()), "background": backgrounds,
           "position": locations1}


def closest_color(rgb):
    r, g, b = rgb
    color_diffs = []
    for color in list(colors.values()):
        cr, cg, cb = color
        color_diff = sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))
    c = min(color_diffs)[1]
    return list(colors.keys())[list(colors.values()).index(c)]

### Image analysis
def detect_contour(img):
    centroid_x, centroid_y = None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if i > 0:
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
    c = contours[-1] if len(contours) > 0 else None
    return centroid_x, centroid_y, c

def detect_shape(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = None
    for temp in shapetemplates:
        template = cv2.imread(temp, 0)
        w, h = template.shape[::-1]
        if img_gray.mean() < 100:
            img_gray = np.array([255 - xi for xi in img_gray])
            img_gray[img_gray < 200] *= 0
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8 if not "spiral" in temp else 0.5
        loc = np.where(res >= threshold)
        if (not loc[0].size == 0) and (not loc[1].size == 0):
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            #cv2.imshow('shapes', img_gray)
            #cv2.waitKey(1000)
            #cv2.destroyAllWindows()
            shape = os.path.basename(temp).split("_")[0]
    return shape

def detect_background_color(img):
    x, y, contour = detect_contour(img)
    color_name = None
    if contour is not None and img.mean() > 150:
        mask = np.ones(img.shape[:2], np.uint8) * 255
        cv2.drawContours(mask, contour, -1, 0, -1)
        cv2.fillPoly(mask, pts=[contour], color=(0, 0, 0))
        mean_color = cv2.mean(img, mask=mask)
        color_name = closest_color(list(mean_color[:3]))
    elif img.mean() < 150:
        color_name = "black"
    return "on " + color_name

def detect_shape_size(img):
    x, y, contour = detect_contour(img)
    size = None
    if contour is not None:
        mask = np.ones(img.shape[:2], np.uint8) * 255
        cv2.drawContours(mask, contour, -1, 0, -1)
        cv2.fillPoly(mask, pts=[contour], color=(0, 0, 0))
        if np.count_nonzero(mask == 255) > 0:
            shape_fraction = np.count_nonzero(mask == 0) / np.count_nonzero(mask == 255)
            size = "large" if shape_fraction > 0.16 else "small"
    return size

def detect_shape_pos(img):
    pos, x, y = None, None, None
    x, y, _ = detect_contour(img)
    if y is not None and y<32:
        pos = "at the top"
    elif y is not None and y>=32:
        pos = "at the bottom"
    if x is not None and x<32:
        pos += " left"
    elif x is not None and x>=32:
        pos += " right"
    return pos

def detect_shape_color(img):
    x, y, contour = detect_contour(img)
    color_name = None
    if contour is not None:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, contour, -1, 255, -1)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
        mean_color = cv2.mean(img, mask=mask)
        color_name = closest_color(list(mean_color[:3]))
    return color_name

def detect_attribute(attribute, img):
    detect_function = {"shape":detect_shape, "size":detect_shape_size, "color":detect_shape_color,
                       "background":detect_background_color, "position":detect_shape_pos}
    attribute_val = detect_function[attribute](img)
    return attribute_val

def image_has_attribute_val(attribute, val, img):
    real_att_val = detect_attribute(attribute, img)
    if real_att_val is not None:
        if real_att_val.lower() == val.lower():
            return True
    return False


### text analysis
def find_in_list(target, source):
    for i in target:
        if i.lower() in source.lower():
            return i.lower()
    return None

def get_attribute(attribute, txt):
    attribute = find_in_list(sources[attribute], txt)
    if attribute is not None:
        return attribute
    else:
        pass # TODO

def search_att(txt, source, idx=None, indices=None):
    att = None
    try:
        for s in source:
            if idx is not None:
                inp = txt.split(" ")[idx]
            elif indices is not None:
                inp = " ".join([txt.split(" ")[i] for i in indices])
            att = find_in_list([s], inp)
            if att is not None:
                return att
    except:
        att = None
    return att

def get_attribute_from_recon(attribute, txt):
    source = sources[attribute]
    if attribute == "size":
        idx, indices = 0, None
    elif attribute == "shape":
        idx, indices = {1:0, 2:1, 3:2, 4:2, 5:2}[difflevel], None
    elif attribute == "color":
        idx, indices = {3:1, 4:1, 5:1}[difflevel], None
    elif attribute == "background":
        idx, indices = None, [-2, -1]
    else:
        idx, indices = None, [3,4,5,6]
    att = search_att(txt, source, idx=idx, indices=indices)
    return att

def try_retrieve_atts(txt):
    atts = []
    for a in level_attributes[difflevel]:
        a = get_attribute_from_recon(a, txt)
        if a is None:
            return None
        else:
            atts.append(a)
    return " ".join(atts)

def load_images(path):
    images = (glob.glob(os.path.join(path, "*.png")))
    images = sorted(images)
    dataset = np.zeros((len(images), 64, 64, 3), dtype=np.float)
    for i, image_path in enumerate(images):
        image = imageio.imread(image_path)
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
        dataset[i, :] = image
    return dataset

def preprocess_image(img):
    return Image.fromarray(img)

def check_cross_sample_correct(testimage=None, testtext=None, reconimage=None, recontext=None):
    correct_attributes = []
    if testimage is not None:
        assert testtext is None, "cross accuracy can be computed only on single input modality"
    if testtext is not None:
        assert testimage is None, "cross accuracy can be computed only on single input modality"
        for att in level_attributes[difflevel]:
            att_value = get_attribute(att, testtext)
            correct = image_has_attribute_val(att, att_value, reconimage)
            correct_attributes.append(int(correct))
    if sum(correct_attributes) ==  len(level_attributes[difflevel]):
        return True
    return False

def calculate_cross_coherency(model):
    with open("./eval/templates/cross_level{}.pkl".format(difflevel), 'rb') as handle:
        t = pickle.load(handle)
    acc_all = []
    for x in range(10):
        images, texts = text_to_image(t, model, args.model)
        correct_pairs = []
        for ind, img in enumerate(images):
            img = np.asarray(img, dtype="uint8")
            correct = check_cross_sample_correct(testtext=t[ind], reconimage=img)
            correct_pairs.append(int(correct))
        acc = sum(correct_pairs) / len(correct_pairs)
        acc_all.append(acc)
        #print("Cross-generation accuracy is {} %".format(acc*100))
    acc_mean = sum(acc_all)/len(acc_all)
    print("Mean cross-generation accuracy is {} %".format(acc_mean * 100))
    return acc_mean


def calculate_joint_coherency(model):
    samples = get_traversal_samples(latent_dim=model.n_latents, n_samples_per_dim=10)
    recons = model.generate_from_latents(samples.unsqueeze(0).cuda())
    img_recons = recons[0].loc
    txt_recons = recons[1].loc
    correct_pairs = []
    for ind, txt in enumerate(txt_recons):
        text = output_onehot2text(recon=txt.unsqueeze(0))[0][0]
        atts = try_retrieve_atts(text)
        if atts is not None:
            img = np.asarray(img_recons[ind].detach().cpu()*255, dtype="uint8")
            correct = check_cross_sample_correct(testtext=atts, reconimage=img)
        else:
            correct = 0
        correct_pairs.append(int(correct))
    acc = sum(correct_pairs) / len(correct_pairs)
    print("Joint-generation accuracy is {} %".format(acc*100))
    return acc

def listdirs(rootdir):
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def labelled_tsne(model):
    testset, testset_len = trainer.prepare_testset(num_samples=250)
    labels = []
    for p in level_attributes[args.level]:
        lp = "./data/level{}/{}_labels.pkl".format(args.level, p)
        with open(lp, 'rb') as handle:
            d = pickle.load(handle)
            labels.append(d)
    for i, label in enumerate(labels):
        lrange = int(len(label) * (1 - config["test_split"]))
        model.analyse(testset, trainer.mPath, i+9999, label[lrange:lrange + testset_len])
    print("Saved labelled T-SNEs in {}".format(os.path.join(trainer.mPath, "visuals")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="path to the model directory")
    parser.add_argument("-l", "--level", type=int, help="difficulty level: 1-5")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    difflevel = args.level
    model, config = load_model(args.model)
    trainer = Trainer(config, dev)
    model = model
    labelled_tsne(model)
    calculate_cross_coherency(model)
    calculate_joint_coherency(model)





