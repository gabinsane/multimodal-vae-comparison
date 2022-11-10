# code for qualitative evaluation of multimodal VAEs trained on GeBiD dataset
import argparse
import glob, yaml
import imageio
import statistics as stat
import cv2
import numpy as np
import pickle
import torch
import os, glob
from utils import listdirs, last_letter
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import output_onehot2text, one_hot_encode

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


def get_attribute_from_recon(attribute, txt, m_exp):
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
        idx, indices = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2}[m_exp.level], None
    elif attribute == "color":
        idx, indices = {3: 1, 4: 1, 5: 1}[m_exp.level], None
    elif attribute == "background":
        idx, indices = None, [-2, -1]
    else:
        idx, indices = None, [3, 4, 5, 6]
    att = search_att(txt, source, idx=idx, indices=indices)
    return att


def try_retrieve_atts(txt, m_exp):
    """

    :param txt:
    :type txt:
    :return:
    :rtype:
    """
    atts = []
    for a in level_attributes[m_exp.level]:
        a = get_attribute_from_recon(a, txt, m_exp)
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


def check_cross_sample_correct(testtext, m_exp, reconimage=None, recontext=None):
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
    for att in level_attributes[m_exp.level]:
        correct = 0
        if reconimage is not None:
            att_value = get_attribute(att, testtext)
            if att_value:
                correct = image_has_attribute_val(att, att_value, reconimage)
        elif recontext is not None:
            attr = get_attribute_from_recon(att, recontext, m_exp)
            if attr is not None:
                if attr in testtext:
                    correct = 1
        correct_attributes.append(int(correct))
    correct_features = sum(correct_attributes) / len(level_attributes[m_exp.level])
    if sum(correct_attributes) == len(level_attributes[m_exp.level]):
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

def text_to_image(text, model_exp):
    """
    Reconstructs text from the image input using the provided model
    :param text: list of strings to reconstruct
    :type text: list
    :param model: model object
    :type model: object
    :param path: where to save the outputs
    :type path: str
    :return: returns reconstructed images and also texts
    :rtype: tuple(list, list)
    """
    img_outputs, txtoutputs = [], []
    for i, w in enumerate(text):
        txt_inp = one_hot_encode(len(w), w.lower())
        inp = {"mod_1":{"data":None, "masks":None}, "mod_2":{"data":txt_inp.unsqueeze(0).to(torch.device("cuda")), "masks":None}}
        recons = model_exp.model.to(torch.device("cuda")).forward(inp)
        recons1, recons2 = recons.mods["mod_1"].decoder_dist.loc[0], recons.mods["mod_2"].decoder_dist.loc
        image, recon_text = model_exp.datamodule.datasets[0].get_processed_recons(recons1), \
                            model_exp.datamodule.datasets[1].get_processed_recons(recons2)
        txtoutputs.append(recon_text[0])
        img_outputs.append(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(os.path.join(path, "cross_sample_{}_{}.png".format(recon_text[0][0][:len(w)], i)), image)
    return img_outputs, txtoutputs

def image_to_text(imgs, model_exp):
        """
        Reconstructs image from the text input using the provided model
        :param imgs: list of images to reconstruct
        :type imgs: list
        :param model: model object
        :type model: object
        :param path: where to save the outputs
        :type path: str
        :return: returns reconstructed images and texts
        :rtype: tuple(list, list)
        """
        txt_outputs, img_outputs = [], []
        for i, w in enumerate(imgs):
            inp = {"mod_1": {"data": w.unsqueeze(0).to(torch.device("cuda")), "masks": None},
                   "mod_2": {"data": None, "masks": None}}
            recons = model_exp.model.to(torch.device("cuda")).forward(inp)
            recons1, recons2 = recons.mods["mod_1"].decoder_dist.loc[0], recons.mods["mod_2"].decoder_dist.loc
            image, recon_text = model_exp.datamodule.datasets[0].get_processed_recons(recons1),  model_exp.datamodule.datasets[1].get_processed_recons(recons2)
            txt_outputs.append(recon_text[0])
            img_outputs.append(image)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(os.path.join(path, "cross_sample_{}_{}.png".format(recon_text[0][0][:len(w)], i)), image)
        return img_outputs, txt_outputs


def calculate_cross_coherency(model_exp):
    """
    Calculates the cross-coherency accuracy for the given model (Img -> Txt and Txt -> Img)

    :param model: multimodal VAE
    :type model: object
    :return: mean cross accuracies
    :rtype: dict
    """
    with open("./eval/templates/cross_level_{}.pkl".format(model_exp.level), 'rb') as handle:
        t = pickle.load(handle)[:100]
        if model_exp.level > 1:
            t = [" ".join(x) for x in t]
    with open("./eval/templates/cross_level{}_images.pkl".format(model_exp.level), 'rb') as handle:
        test_images = pickle.load(handle)[:100]
        test_images = torch.tensor(test_images).permute(0, 3, 2, 1) / 255
    acc_all = {"text_image": [0, 0], "image_text": [0, 0, 0]}
    images, texts = text_to_image(t, model_exp)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, img in enumerate(images):
        img = np.asarray(img, dtype="uint8")
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], m_exp=model_exp, reconimage=img)
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
        corr_letters.append(corr_letter)
    acc_all["text_image"] = get_mean_stats([correct_pairs, corr_feats])
    images, texts = image_to_text(test_images, model_exp)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, txt in enumerate(texts):
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], m_exp=model_exp, recontext=txt)
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
    recons = model_exp.save_joint_samples(num_samples=32, savedir=os.path.join(model_exp.config.mPath, "visuals/"))
    img_recons = recons["mod_1"]
    txt_recons = recons["mod_2"]
    correct_pairs, corr_feats = [], []
    for ind, txt in enumerate(txt_recons):
        atts = try_retrieve_atts(txt, model_exp)
        if atts is not None:
            img = np.asarray(img_recons[ind], dtype="uint8")
            correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=atts, m_exp=model_exp, reconimage=img)
        else:
            correct = 0
            corr_feat = 0
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
    output = {"joint": get_mean_stats([correct_pairs, corr_feats])}
    return output


def prepare_labels(labels, model_exp):
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
    for i in range(model_exp.level):
        if model_exp.level == 1:
            ls = labels_cropped
        elif model_exp.level in [2, 3]:
            ls = [x.split(" ")[i] for x in labels_cropped]
        elif model_exp.level == 4:
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
    testset, _ = model_exp.datamodule.get_num_samples(250, split="test")
    labels = prepare_labels(testset["mod_2"], model_exp)
    for i, label in enumerate(labels):
        model_exp.analyse_data(testset, label, path_name="eval_{}".format(i), savedir=os.path.join(model_exp.config.mPath, "visuals"))
    print("Saved labelled T-SNEs in {}".format(os.path.join(model_exp.config.mPath, "visuals")))



def eval_all(model_exp):
    print("- Making T-SNE for each feature")
    labelled_tsne(model_exp)
    print("- Calculating cross coherency")
    output_cross = calculate_cross_coherency(model_exp)
    print("- Calculating joint coherency")
    output_joint = calculate_joint_coherency(model_exp)
    return output_cross, output_joint

def print_save_stats(stats_dict, path):
    print("Final results:")
    with open(os.path.join(path,'gebid_stats.txt'), 'w') as f:
        for key, value_dict in stats_dict.items():
            if value_dict["stdev"] is not None:
                stat = "{}: {:.2f} ({:.2f})".format(key, round(value_dict["value"],2), round(value_dict["stdev"], 2))
            else:
                stat = "{}: {:.2f}".format(key, round(value_dict["value"], 2))
            print(stat)
            f.write(stat)
            f.write('\n')
    print("\n GeBiD statistics printed in {} \n".format(os.path.join(path,'gebid_stats.txt')))

def eval_single_model(m_exp):
    print("\nCalculating GeBiD automatic statistics")
    level = int(os.path.dirname(m_exp.config.mods[0]["path"])[-1])
    m_exp.level = level
    output_cross, output_joint = eval_all(m_exp)
    output_dict = {"Text-Image Strict":{"value":output_cross["text_image"][0], "stdev":None},
                   "Text-Image Features":{"value":output_cross["text_image"][1], "stdev":None},
                   "Image-Text Strict":{"value":output_cross["image_text"][0], "stdev":None},
                   "Image-Text Features":{"value":output_cross["image_text"][1], "stdev":None},
                   "Image-Text Letters":{"value":output_cross["image_text"][2], "stdev":None},
                   "Joint Strict":{"value":output_joint["joint"][0], "stdev":None},
                   "Joint Features":{"value":output_joint["joint"][1], "stdev":None}}
    print_save_stats(output_dict, m_exp.config.mPath)

def fill_cats(text_image, image_text, joint, data):
    for i, x in enumerate(data["text_image"]):
        text_image[i].append(x)
    for i, x in enumerate(data["image_text"]):
        image_text[i].append(x)
    for i, x in enumerate(data["joint"]):
        joint[i].append(x)
    return text_image, image_text, joint

def eval_gebid_over_seeds(parent_dir):
    all_models = listdirs(parent_dir)
    all_models = sorted(all_models, key=last_letter)
    text_image = [[], []]
    image_text = [[], [], []]
    joint = [[], []]
    for m in all_models:
        if os.path.exists(os.path.join(m, "gebid_stats.txt")):
            print("Model: {} already has the statistics".format(m))
            with open(os.path.join(m, "gebid_stats.txt"), 'r') as stream:
                d = yaml.safe_load(stream)
                ls = [["Text-Image", "Image-Text", "Joint"], ["Strict", "Features", "Letters"]]
                output = {}
                for a in ls[0]:
                    key = a.lower().replace("-", "_")
                    output[key] = []
                    for b in ls[1]:
                        valname =  " ".join((a, b))
                        if valname in d.keys():
                            value = d[valname] if not isinstance(d[valname], str) else float(d[valname].split(" (")[0])
                            output[key].append(value)
        else:
            latest_version = max(glob.glob(os.path.join(m, "lightning_logs/" '*/')), key=os.path.getmtime)
            latest_ckpt = max(glob.glob(os.path.join(latest_version, "checkpoints",'*')), key=os.path.getmtime)
            print("Model: {}".format(latest_ckpt))
            exp = MultimodalVAEInfer(latest_ckpt)
            model = exp.get_wrapped_model()
            eval_single_model(model)
            output, output_joint = eval_all(model)
            output["joint"] = output_joint["joint"]
        text_image, image_text, joint = fill_cats(text_image, image_text, joint, output)
    output_dict = {"Text-Image Strict":{"value":stat.mean(text_image[0]), "stdev":stat.stdev(text_image[0])},
                   "Text-Image Features":{"value":stat.mean(text_image[1]), "stdev":stat.stdev(text_image[1])},
                   "Image-Text Strict":{"value":stat.mean(image_text[0]), "stdev":stat.stdev(image_text[0])},
                   "Image-Text Features":{"value":stat.mean(image_text[1]), "stdev":stat.stdev(image_text[1])},
                   "Image-Text Letters":{"value":stat.mean(image_text[2]), "stdev":stat.stdev(image_text[2])},
                   "Joint Strict":{"value":stat.mean(joint[0]), "stdev":stat.stdev(joint[0])},
                   "Joint Features":{"value":stat.mean(joint[1]), "stdev":stat.stdev(joint[1])}}
    print_save_stats(output_dict, parent_dir)


if __name__ == "__main__":
    from eval.infer import MultimodalVAEInfer
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mpath", type=str, help="path to the .ckpt model file. Relative or absolute")
    parser.add_argument("-m", "--multieval", type=str, help="path to parent directory with mutliple models")
    args = parser.parse_args()
    assert not (args.mpath and args.multieval), "You can only provide one of these arguments: mpath or mutlieval (not both)"
    if args.mpath:
        exp = MultimodalVAEInfer(args.mpath)
        model = exp.get_wrapped_model()
        eval_single_model(model)
    else:
        eval_gebid_over_seeds(args.multieval)
