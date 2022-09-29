# code for qualitative evaluation of multimodal VAEs trained on GeBiD dataset
import argparse
import glob
import imageio
import statistics as stat

import cv2
import numpy as np
import pickle
import torch
import warnings
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import output_onehot2text
from eval.infer import MMVAEExperiment, define_path

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
        idx, indices = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2}[args.level], None
    elif attribute == "color":
        idx, indices = {3: 1, 4: 1, 5: 1}[args.level], None
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
    for a in level_attributes[args.level]:
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
    for att in level_attributes[args.level]:
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
    correct_features = sum(correct_attributes) / len(level_attributes[args.level])
    if sum(correct_attributes) == len(level_attributes[args.level]):
        all_correct = True
    else:
        all_correct = False
    if recontext is not None:
        corr_letters = count_same_letters(recontext, testtext)
        correct_letters = corr_letters / len(testtext)
        all_correct = 1 if correct_letters == 1 else 0
    return all_correct, correct_features, correct_letters


def get_mean_stats(list_of_stats, percentage=True):
    """
    Returns a list of means for a nested list with accuracies

    :param list_of_stats: multiple lists with accuracies
    :type list_of_stats: list
    :param percentage: whether to report the number as percent (True) or fraction (False)
    :type percentage: bool
    :return: a list of means of the accuracies
    :rtype: list
    """
    stats = []
    for l in list_of_stats:
        if percentage:
            stats.append(100 * sum(l) / len(l))
        else:
            stats.append(sum(l) / len(l))
    return stats


def calculate_cross_coherency(model_exp):
    """
    Calculates the cross-coherency accuracy for the given model (Img -> Txt and Txt -> Img)

    :param model: multimodal VAE
    :type model: object
    :return: mean cross accuracies
    :rtype: dict
    """
    with open("./eval/templates/cross_level_{}.pkl".format(args.level), 'rb') as handle:
        t = pickle.load(handle)[:100]
        if args.level > 1:
            t = [" ".join(x) for x in t]
    with open("./eval/templates/cross_level{}_images.pkl".format(args.level), 'rb') as handle:
        test_images = pickle.load(handle)[:100]
        test_images = torch.tensor(test_images).permute(0, 3, 2, 1) / 255
    acc_all = {"text_image": [0, 0], "image_text": [0, 0, 0]}
    images, texts = model_exp.text_to_image(t)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, img in enumerate(images):
        img = np.asarray(img, dtype="uint8")
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], reconimage=img)
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
        corr_letters.append(corr_letter)
    acc_all["text_image"] = get_mean_stats([correct_pairs, corr_feats])
    images, texts = model_exp.image_to_text(test_images)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, txt in enumerate(texts):
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], recontext=txt)
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
        corr_letters.append(corr_letter)
    acc_all["image_text"] = get_mean_stats([correct_pairs, corr_feats, corr_letters])
    return acc_all


def calculate_joint_coherency(model_exp):
    """
    Calculates the joint-coherency accuracy for the given model

    :param model: multimodal VAE
    :type model: object
    :return: mean joint accuracy
    :rtype: dict
    """
    samples = model_exp.get_traversal_samples(latent_dim=model_exp.model.config.n_latents, n_samples_per_dim=15)
    recons = model_exp.decode_latents(samples)
    img_recons = recons["mod_1"][0][0]
    txt_recons = recons["mod_2"][0][0]
    correct_pairs, corr_feats = [], []
    for ind, txt in enumerate(txt_recons):
        text = output_onehot2text(recon=txt.unsqueeze(0))[0]
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
    l = output_onehot2text(labels["data"])
    labels_cropped = []
    for v_i, v in enumerate(l):
        o = [s for n, s in enumerate(v) if labels["masks"][v_i][n]]
        labels_cropped.append("".join(o))
    labels_sep = []
    for i in range(args.level):
        if args.level == 1:
            ls = labels_cropped
        elif args.level in [2, 3]:
            ls = [x.split(" ")[i] for x in labels_cropped]
        elif args.level == 4:
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


def labelled_tsne(model_exp):
    """
    Plots the T-SNE visualizations with the GeBiD text labels (one T-SNE per each feature)

    :param model: multimodal VAE
    :type model: object
    """
    model_exp.get_test_data_bs(batch_size=250)
    testset = model_exp.get_test_data_sample()
    labels = prepare_labels(testset["mod_2"])
    for i, label in enumerate(labels):
        model_exp.model.analyse_data(testset, label, path_label="eval_{}".format(i))
    print("Saved labelled T-SNEs in {}".format(os.path.join(model_exp.get_base_path(), "visuals")))


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


def eval_all(model_exp):
    labelled_tsne(model_exp)
    output_cross = calculate_cross_coherency(model_exp)
    output_joint = calculate_joint_coherency(model_exp)
    return output_cross, output_joint

def print_save_stats(stats_dict, path):
    with open(os.path.join(os.path.dirname(path),'gebid_stats.txt'), 'w') as f:
        for key, value_dict in stats_dict.items():
            if value_dict["stdev"]:
                stat = "{}: {:.2f} ({:. 2f})".format(key, round(value_dict["value"],2), round(value_dict["stdev"], 2))
            else:
                stat = "{}: {:.2f}".format(key, round(value_dict["value"], 2))
            print(stat)
            f.write(stat)
            f.write('\n')


def eval_single_model(pth):
    m_exp = MMVAEExperiment(path=pth)
    m_exp.model.to(torch.device("cuda"))
    output_cross, output_joint = eval_all(m_exp)
    output_dict = {"Text-Image Strict":{"value":output_cross["text_image"][0], "stdev":None},
                   "Text-Image Features":{"value":output_cross["text_image"][1], "stdev":None},
                   "Image-Text Strict":{"value":output_cross["image_text"][0], "stdev":None},
                   "Image-Text Features":{"value":output_cross["image_text"][1], "stdev":None},
                   "Image-Text Letters":{"value":output_cross["image_text"][2], "stdev":None},
                   "Joint Strict":{"value":output_joint["joint"][0], "stdev":None},
                   "Joint Features":{"value":output_joint["joint"][1], "stdev":None}}
    print_save_stats(output_dict, pth)



def eval_over_seeds(parent_dir):
    all_models = listdirs(parent_dir)
    all_models = sorted(all_models, key=last_letter)
    text_image = [[], []]
    image_text = [[], [], []]
    joint = [[], []]
    for m in all_models:
        pth = os.path.join(m, "last.ckpt")
        m_exp = MMVAEExperiment(path=pth)
        m_exp.model.to(torch.device("cuda"))
        output_cross, output_joint = eval_all(m_exp)
        for i, x in enumerate(output_cross["text_image"]):
              text_image[i].append(x)
        for i, x in enumerate(output_cross["image_text"]):
                 image_text[i].append(x)
        for i, x in enumerate(output_joint["joint"]):
                 joint[i].append(x)
    output_dict = {"Text-Image Strict":{"value":stat.mean(text_image[0]), "stdev":stat.stdev(text_image[0])},
                   "Text-Image Features":{"value":stat.mean(text_image[1]), "stdev":stat.stdev(text_image[1])},
                   "Image-Text Strict":{"value":stat.mean(image_text[0]), "stdev":stat.stdev(image_text[0])},
                   "Image-Text Features":{"value":stat.mean(image_text[1]), "stdev":stat.stdev(image_text[1])},
                   "Image-Text Letters":{"value":stat.mean(image_text[2]), "stdev":stat.stdev(image_text[2])},
                   "Joint Strict":{"value":stat.mean(joint[0]), "stdev":stat.stdev(joint[0])},
                   "Joint Featres":{"value":stat.mean(joint[1]), "stdev":stat.stdev(joint[1])}}
    print_save_stats(output_dict, parent_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="path to the model directory")
    parser.add_argument("-multi", "--model_multi", type=str, help="path to a directory containing multiple trained models. will eval all and produce mean statistics" )
    parser.add_argument("-l", "--level", type=int, help="difficulty level: 1-5", required=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    assert args.model or args.model_multi, "Provide either path to a single model directory or a parent directory " \
                                           "containing multiple models"
    if args.model:
        p = define_path(args.model)
        eval_single_model(p)
    elif args.model_multi:
        assert os.path.exists(args.model_multi) and os.path.isdir(args.model_multi)
        eval_over_seeds(args.model_multi)
