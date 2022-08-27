# code for qualitative evaluation of multimodal VAEs trained on GeBiD dataset
import argparse
import glob
import imageio
import os
import statistics as stat

import cv2
import numpy as np
import pickle
import torch

from main import Trainer
from utils import output_onehot2text
from .infer import load_model, text_to_image, get_traversal_samples, image_to_text

shapetemplates = glob.glob('./eval/templates/*.png')
colors = {"yellow": [255, 255, 0], "red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255],
          "grey": [128, 128, 128], "brown": [105, 0, 0], "purple": [215, 0, 215], "teal": [0, 175, 175],
          "navy": [0, 0, 150], "orange": [255, 140, 0], "beige": [232, 211, 185], "pink": [255, 182, 193],
          "black": [0, 0, 0], "white": [255, 255, 255]}
shapenames = ["spiral", "line", "square", "semicircle", "circle", "pieslice"]
sizes = ["small", "large"]
locations1 = ["at the top left", "at the top right", "at the bottom right", "at the bottom left"]
locations2 = ["left", "right"]
backgrounds = ["on white", "on black"]
level_attributes = {1: ["shape"], 2: ["size", "shape"], 3: ["size", "color", "shape"],
                    4: ["size", "color", "shape", "background"],
                    5: ["size", "color", "shape", "position", "background"]}
sources = {"shape": shapenames, "size": sizes, "color": list(colors.keys()), "background": backgrounds,
           "position": locations1}


def manhattan_distance(a, b):
    """
    Calculates the Manharran distance between two vectors

    :param a: vec 1
    :type a: tuple
    :param b: vec 2
    :type b: tuple
    :return: distance
    :rtype: float
    """
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def closest_color(rgb):
    """

    :param rgb:
    :type rgb:
    :return:
    :rtype:
    """
    r, g, b = rgb
    color_diffs = []
    for color in list(colors.values()):
        cr, cg, cb = color
        color_diff = manhattan_distance((r, g, b), (cr, cg, cb))  # sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))
    c = min(color_diffs)[1]
    if c == [128, 128, 128]:
        if stat.stdev((r, g, b)) > 20:  # most probably not gray
            c = sorted(color_diffs)[1][1]
    if list(colors.keys())[list(colors.values()).index(c)] == "pink":
        if b < 170:
            c = colors["orange"]
    return list(colors.keys())[list(colors.values()).index(c)]


# Image analysis
def detect_contour(img):
    """

    :param img:
    :type img:
    :return:
    :rtype:
    """
    centroid_x, centroid_y = None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if not (i == 0):
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
    c = contours[-1] if len(contours) > 0 else None
    return centroid_x, centroid_y, c


def detect_shape(img):
    """

    :param img:
    :type img:
    :return:
    :rtype:
    """
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
            # cv2.imshow('shapes', img_gray)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            shape = os.path.basename(temp).split("_")[0]
    return shape


def detect_shape_color(img):
    """

    :param img:
    :type img:
    :return:
    :rtype:
    """
    x, y, contour = detect_contour(img)
    color_name = None
    if contour is not None:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, contour, -1, 255, -1)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
        mean_color = cv2.mean(img, mask=mask)
        color_name = closest_color(list(mean_color[:3]))
    return color_name


def detect_background_color(img):
    """

    :param img:
    :type img:
    :return:
    :rtype:
    """
    x, y, contour = detect_contour(img)
    color_name = "None"
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
    """

    :param img:
    :type img:
    :return:
    :rtype:
    """
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
    """

    :param img:
    :type img:
    :return:
    :rtype:
    """
    pos, x, y = None, None, None
    x, y, _ = detect_contour(img)
    if y is not None and y < 32:
        pos = "at the top"
    elif y is not None and y >= 32:
        pos = "at the bottom"
    if x is not None and x < 32:
        pos += " left"
    elif x is not None and x >= 32:
        pos += " right"
    return pos


def detect_attribute(attribute, img):
    """

    :param attribute:
    :type attribute:
    :param img:
    :type img:
    :return:
    :rtype:
    """
    detect_function = {"shape": detect_shape, "size": detect_shape_size, "color": detect_shape_color,
                       "background": detect_background_color, "position": detect_shape_pos}
    attribute_val = detect_function[attribute](img)
    return attribute_val


def image_has_attribute_val(attribute, val, img):
    """

    :param attribute:
    :type attribute:
    :param val:
    :type val:
    :param img:
    :type img:
    :return:
    :rtype:
    """
    real_att_val = detect_attribute(attribute, img)
    if real_att_val is not None:
        if real_att_val.lower() == val.lower():
            return True
    return False


### text analysis
def find_in_list(target, source):
    """

    :param target:
    :type target:
    :param source:
    :type source:
    :return:
    :rtype:
    """
    for i in target:
        if i.lower() in source.lower():
            return i.lower()
    return None


def get_attribute(attribute, txt):
    """

    :param attribute:
    :type attribute:
    :param txt:
    :type txt:
    :return:
    :rtype:
    """
    attribute = find_in_list(sources[attribute], txt)
    if attribute is not None:
        return attribute
    else:
        return None


def search_att(txt, source, idx=None, indices=None):
    """

    :param txt:
    :type txt:
    :param source:
    :type source:
    :param idx:
    :type idx:
    :param indices:
    :type indices:
    :return:
    :rtype:
    """
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
    """

    :param attribute:
    :type attribute:
    :param txt:
    :type txt:
    :return:
    :rtype:
    """
    source = sources[attribute]
    if attribute == "size":
        idx, indices = 0, None
    elif attribute == "shape":
        idx, indices = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2}[difflevel], None
    elif attribute == "color":
        idx, indices = {3: 1, 4: 1, 5: 1}[difflevel], None
    elif attribute == "background":
        idx, indices = None, [-2, -1]
    else:
        idx, indices = None, [3, 4, 5, 6]
    att = search_att(txt, source, idx=idx, indices=indices)
    return att


def try_retrieve_atts(txt):
    """

    :param txt:
    :type txt:
    :return:
    :rtype:
    """
    atts = []
    for a in level_attributes[difflevel]:
        a = get_attribute_from_recon(a, txt)
        if a is None:
            atts.append("Unknown")
        else:
            atts.append(a)
    return " ".join(atts)


def load_images(path):
    """
    Loads .png images from a dir path

    :param path: path to the folder
    :type path: str
    :return: list of ndarrays
    :rtype: list
    """
    images = (glob.glob(os.path.join(path, "*.png")))
    images = sorted(images)
    dataset = np.zeros((len(images), 64, 64, 3), dtype=np.float)
    for i, image_path in enumerate(images):
        image = imageio.imread(image_path)
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
        dataset[i, :] = image
    return dataset


def count_same_letters(a, b):
    """
    Counts how many characters are the same in two strings.

    :param a: string 1
    :type a: str
    :param b: string 2
    :type b: str
    :return: number of matching characters
    :rtype: int
    """
    if len(a) > len(b):
        a = a[:len(b)]
    if len(b) > len(a):
        b = b[:len(a)]
    return sum(a[i] == b[i] for i in range(len(a)))


def check_cross_sample_correct(testtext, reconimage=None, recontext=None):
    """
    Detects the features in images/text and checks if they are coherent

    :param testtext: ground truth text input
    :type testtext: str
    :param reconimage: reconstructed image
    :type reconimage: ndarray
    :param recontext: reconstructed text
    :type recontext: str
    :return: returns whether the sample is completely correct, how many features are ok, how many letters are ok
    :rtype: tuple(Bool, float32, float32)
    """
    correct_attributes = []
    correct_letters = None
    assert (reconimage is None) or (recontext is None), "for evaluation of both text and image, use joint_coherency"
    for att in level_attributes[difflevel]:
        correct = 0
        if reconimage is not None:
            att_value = get_attribute(att, testtext)
            if att_value:
                correct = image_has_attribute_val(att, att_value, reconimage)
        elif recontext is not None:
            attr = get_attribute_from_recon(att, recontext)
            if attr is not None:
                if attr in testtext:
                    correct = 1
        correct_attributes.append(int(correct))
    correct_features = sum(correct_attributes) / len(level_attributes[difflevel])
    if sum(correct_attributes) == len(level_attributes[difflevel]):
        all_correct = True
    else:
        all_correct = False
    if recontext is not None:
        corr_letters = count_same_letters(recontext, testtext)
        correct_letters = corr_letters / len(testtext)
        all_correct = 1 if correct_letters == 1 else 0
    return all_correct, correct_features, correct_letters


def get_mean_stats(list_of_stats):
    """
    Returns a list of means for a nested list with accuracies

    :param list_of_stats: multiple lists with accuracies
    :type list_of_stats: list
    :return: a list of means of the accuracies
    :rtype: list
    """
    stats = []
    for l in list_of_stats:
        stats.append(sum(l) / len(l))
    return stats


def calculate_cross_coherency(model):
    """
    Calculates the cross-coherency accuracy for the given model (Img -> Txt and Txt -> Img)

    :param model: multimodal VAE
    :type model: object
    :return: mean cross accuracies
    :rtype: dict
    """
    with open("./eval/templates/cross_level_{}.pkl".format(difflevel), 'rb') as handle:
        t = pickle.load(handle)[:100]
        if difflevel > 1:
            t = [" ".join(x) for x in t]
    with open("./eval/templates/cross_level{}_images.pkl".format(difflevel), 'rb') as handle:
        test_images = pickle.load(handle)[:100]
        test_images = torch.tensor(test_images).permute(0, 3, 2, 1) / 255
    acc_all = {"text_image": [0, 0], "image_text": [0, 0, 0]}
    images, texts = text_to_image(t, model, m)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, img in enumerate(images):
        img = np.asarray(img, dtype="uint8")
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], reconimage=img)
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
        corr_letters.append(corr_letter)
    acc_all["text_image"] = get_mean_stats([correct_pairs, corr_feats])
    images, texts = image_to_text(test_images, model, m)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, txt in enumerate(texts):
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], recontext=txt)
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
        corr_letters.append(corr_letter)
    acc_all["image_text"] = get_mean_stats([correct_pairs, corr_feats, corr_letters])
    return acc_all


def calculate_joint_coherency(model):
    """
    Calculates the joint-coherency accuracy for the given model

    :param model: multimodal VAE
    :type model: object
    :return: mean joint accuracy
    :rtype: dict
    """
    samples = get_traversal_samples(latent_dim=model.n_latents, n_samples_per_dim=15)
    recons = model.generate_from_latents(samples.unsqueeze(0).cuda())
    img_recons = recons[0].loc
    txt_recons = recons[1].loc
    correct_pairs, corr_feats = [], []
    for ind, txt in enumerate(txt_recons):
        text = output_onehot2text(recon=txt.unsqueeze(0))[0][0]
        atts = try_retrieve_atts(text)
        if atts is not None:
            img = np.asarray(img_recons[ind].detach().cpu() * 255, dtype="uint8")
            correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=atts, reconimage=img)
        else:
            correct = 0
            corr_feat = 0
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
    output = {"joint": get_mean_stats([correct_pairs, corr_feats])}
    return output


def prepare_labels(labels):
    """
    Turns the text modality into labels by separating the features.

    :param labels: one-hot text encodings and boolean masks with string lengths
    :type labels: list
    :return: labels for T-SNE
    :rtype: list
    """
    l = output_onehot2text(torch.stack(labels[0]))
    labels_cropped = []
    for v_i, v in enumerate(l[0]):
        o = [s for n, s in enumerate(v) if labels[1][v_i][n]]
        labels_cropped.append("".join(o))
    labels_sep = []
    for i in range(difflevel):
        if difflevel == 1:
            ls = labels_cropped
        elif difflevel in [2, 3]:
            ls = [x.split(" ")[i] for x in labels_cropped]
        elif difflevel == 4:
            if i < 3:
                ls = [x.split(" ")[i] for x in labels_cropped]
            else:
                ls = [" ".join(x.split(" ")[-2:]) for x in labels_cropped]
        else:
            if i < 3:
                ls = [x.split(" ")[i] for x in labels_cropped]
            elif i == 3:
                ls = [" ".join(x.split(" ")[-2:]) for x in labels_cropped]
            else:
                ls = [" ".join(x.split(" ")[3:7]) for x in labels_cropped]
        labels_sep.append(ls)
    return labels_sep


def labelled_tsne(model):
    """
    Plots the T-SNE visualizations with the GeBiD text labels (one T-SNE per each feature)

    :param model: multimodal VAE
    :type model: object
    """
    testset, testset_len = trainer.prepare_testset(num_samples=250)
    labels = prepare_labels(testset[1])
    for i, label in enumerate(labels):
        model.analyse(testset, m, "eval_model_{}_{}".format(i, 1), label)
    print("Saved labelled T-SNEs in {}".format(os.path.join(trainer.mPath, "visuals")))


def listdirs(rootdir):
    """
    Lists all the subdirectories within a directory

    :param rootdir: path to the root dir
    :type rootdir: str
    :return: list of subdirectories
    :rtype: str
    """
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def last_letter(word):
    return word[::-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="path to the model directory")
    parser.add_argument("-l", "--level", type=int, help="difficulty level: 1-5")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with torch.no_grad():
        for ll in ["1", "2", "3", "4", "5"]:
            difflevel = int(ll)
            print("LEVEL {}".format(ll))
            for dim in ["32", "64", "128"]:
                print("DIM {}".format(dim))
                for name in ["poe", "moe"]:
                    print("MODEL {}".format(name))
                    all_models = listdirs(
                        "/home/gabi/multimodal-vae-comparison/multimodal_compare/results/results_alllevels/{}/{}/{}".format(
                            ll, dim, name))
                    all_models = sorted(all_models, key=last_letter)
                    text_image = [[], []]
                    image_text = [[], [], []]
                    joint = [[], []]
                    for m in all_models:
                        model = load_model(m)
                        model, config = load_model(m, modelname="model.rar")
                        trainer = Trainer(config, dev)
                        labelled_tsne(model)
                    #     output = calculate_cross_coherency(model)
                    #     #print(output)
                    #     for i, x in enumerate(output["text_image"]):
                    #         text_image[i].append(x)
                    #     for i, x in enumerate(output["image_text"]):
                    #         image_text[i].append(x)
                    #     output = calculate_joint_coherency(model)
                    #     #print(output)
                    #     for i, x in enumerate(output["joint"]):
                    #          joint[i].append(x)
                    # print("Text-Image Whole")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(text_image[0]),2), round(stat.stdev(text_image[0]),2)))
                    # print("Text-Image Features")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(text_image[1]),2), round(stat.stdev(text_image[1]),2)))
                    # print("Image-Text Whole")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(image_text[0]),2), round(stat.stdev(image_text[0]),2)))
                    # print("Image-Text Features")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(image_text[1]),2), round(stat.stdev(image_text[1]),2)))
                    # print("Image-Text Letters")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(image_text[2]),2), round(stat.stdev(image_text[2]),2)))
                    # print("Joint Whole")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(joint[0]),2), round(stat.stdev(joint[0]),2)))
                    # print("Joint Features")
                    # print("{:.2f} ({:.2f})".format(round(stat.mean(joint[1]),2), round(stat.stdev(joint[1]),2)))
