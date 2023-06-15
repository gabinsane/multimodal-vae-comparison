# code for qualitative evaluation of multimodal VAEs trained on GeBiD dataset
import argparse
import glob, yaml
import imageio
import statistics as stat
import cv2
import numpy as np
import torch
from eval.train_classifiers import CNN
import os, glob
from utils import listdirs, last_letter, print_save_stats
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import output_onehot2text, one_hot_encode

colors = {'white': [255, 255, 255],
          'red': [192, 64, 0],
          'yellow': [228, 217, 111],
          'green': [10, 107, 60],
          'blue': [0, 127, 200],
           'pink': [255, 0, 255]}
shapenames = ["heart", "ellipse", "square"]
sizes = ["small", "big"]
locations = ["at top left", "at top right", "at bottom right", "at bottom left"]
backgrounds = ["on light", "on dark"]
level_attributes = {1: ["shape"], 2: ["size", "shape"], 3: ["size", "color", "shape"],
                    4: ["size", "color", "shape", "position"],
                    5: ["size", "color", "shape", "position", "background"]}
sources = {"shape": shapenames, "size": sizes, "color": list(colors.keys()), "background": backgrounds,
           "position": locations}
class_mappings = {"shape": ["square", "ellipse", "heart"], "size": ["big", "small"],
            "color": ["blue", "green", "red", "yellow", "pink"],
            "position": ["at top left", "at top right", "at bottom left", "at bottom right"],
            "background": ["on light", "on dark"]}

def get_all_classifiers(level):
    print("Loading_classifiers...")
    classifiers = {}
    for key in level_attributes[level]:
        classifiers[key] = load_classifier(level, key)
    print("Done.")
    return classifiers

def load_classifier(level:int, class_type:str):
    model = CNN(class_type).to("cuda")
    model.load_state_dict(torch.load("./eval/classifiers/cdspritesplus_classifier_level{}_{}.pth".format(level, class_type)))
    return model

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

def eval_with_classifier(classifier, image, att):
    recognized = torch.argmax(classifier.forward(torch.tensor(image.reshape(-1,3,64,64))/255), dim=-1)
    return class_mappings[att][int(recognized)]

def check_cross_sample_correct(testtext, m_exp, classifiers=None, reconimage=None, recontext=None):
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
                recognized_val = eval_with_classifier(classifiers[att], reconimage, att)
                if att_value == recognized_val:
                    correct = 1
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
        image, recon_text = model_exp.datamod.datasets[0].get_processed_recons(recons1), \
                            model_exp.datamod.datasets[1].get_processed_recons(recons2)
        txtoutputs.append(recon_text[0])
        img_outputs.append(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(model_exp.config.mPath, "cross_sample_{}_{}.png".format(recon_text[0][0][:len(w)], i)), image)
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
            image, recon_text = model_exp.datamod.datasets[0].get_processed_recons(recons1),  model_exp.datamod.datasets[1].get_processed_recons(recons2)
            txt_outputs.append(recon_text[0])
            img_outputs.append(image)
        return img_outputs, txt_outputs

def get_mod_mappings(mod_dict):
    if mod_dict ["mod_1"]["masks"] is None and mod_dict["mod_2"]["masks"] is not None:
        return {"image":"mod_1", "text":"mod_2"}
    elif mod_dict ["mod_1"]["masks"] is not None and mod_dict["mod_2"]["masks"] is None:
        return {"image": "mod_2", "text": "mod_1"}
    else:
        raise Exception("The provided data does not seem to be composed of text and image")

def calculate_cross_coherency(model_exp, classifiers):
    """
    Calculates the cross-coherency accuracy for the given model (Img -> Txt and Txt -> Img)

    :param model: multimodal VAE
    :type model: object
    :return: mean cross accuracies
    :rtype: dict
    """
    testset, labels = model_exp.datamod.get_num_samples(250, split="test")
    mapping = get_mod_mappings(testset)
    t = [" ".join(x) for x in labels] if model_exp.level > 1 else labels
    test_images = testset[mapping["image"]]["data"]
    acc_all = {"text_image": [0, 0], "image_text": [0, 0, 0]}
    images, texts = text_to_image(t, model_exp)
    correct_pairs, corr_feats, corr_letters = [], [], []
    for ind, img in enumerate(images):
        img = np.asarray(img, dtype="uint8")
        correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=t[ind], classifiers=classifiers, m_exp=model_exp, reconimage=img)
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


def calculate_joint_coherency(model_exp, classifiers):
    """
    Calculates the joint-coherency accuracy for the given model

    :param model: multimodal VAE
    :type model: object
    :return: mean joint accuracy
    :rtype: dict
    """
    recons = model_exp.save_joint_samples(num_samples=64, savedir=os.path.join(model_exp.config.mPath, "visuals/"))
    img_recons = recons["mod_1"]
    txt_recons = recons["mod_2"]
    correct_pairs, corr_feats = [], []
    for ind, txt in enumerate(txt_recons):
        atts = try_retrieve_atts(txt, model_exp)
        if atts is not None:
            img = np.asarray(img_recons[ind], dtype="uint8")
            correct, corr_feat, corr_letter = check_cross_sample_correct(testtext=atts, classifiers=classifiers, m_exp=model_exp, reconimage=img)
        else:
            correct = 0
            corr_feat = 0
        correct_pairs.append(int(correct))
        corr_feats.append(corr_feat)
    output = {"joint": get_mean_stats([correct_pairs, corr_feats])}
    return output


def eval_all(model_exp, classifiers):
    print("- Calculating cross coherency")
    output_cross = calculate_cross_coherency(model_exp, classifiers)
    print("- Calculating joint coherency")
    output_joint = calculate_joint_coherency(model_exp, classifiers)
    return output_cross, output_joint

def eval_single_model(m_exp):
    #m_exp.save_reconstructions(3, savedir=os.path.dirname(args.mpath))
    level = int(os.path.dirname(m_exp.config.mods[0]["path"])[-1])
    m_exp.level = level
    classifiers = get_all_classifiers(m_exp.level)
    print("\nCalculating CdSprites+ automatic statistics")
    output_cross, output_joint = eval_all(m_exp, classifiers)
    output_dict = {"Text-Image Strict":{"value":output_cross["text_image"][0], "stdev":None},
                    "Text-Image Features":{"value":output_cross["text_image"][1], "stdev":None},
                    "Image-Text Strict":{"value":output_cross["image_text"][0], "stdev":None},
                    "Image-Text Features":{"value":output_cross["image_text"][1], "stdev":None},
                    "Image-Text Letters":{"value":output_cross["image_text"][2], "stdev":None},
                    "Joint Strict":{"value":output_joint["joint"][0], "stdev":None},
                    "Joint Features":{"value":output_joint["joint"][1], "stdev":None}}
    print_save_stats(output_dict, m_exp.config.mPath, "cdspritesplus", level)

def fill_cats(text_image, image_text, joint, data):
    for i, x in enumerate(data["text_image"]):
        text_image[i].append(x)
    for i, x in enumerate(data["image_text"]):
        image_text[i].append(x)
    for i, x in enumerate(data["joint"]):
        joint[i].append(x)
    return text_image, image_text, joint

def eval_cdsprites_over_seeds(parent_dir):
    all_models = listdirs(parent_dir)
    all_models = sorted(all_models, key=last_letter)
    text_image = [[], []]
    image_text = [[], [], []]
    joint = [[], []]
    classifiers = None
    for m in all_models:
        output = None
        if os.path.exists(os.path.join(m, "cdspritesplus_stats.txt")):
            print("Model: {} already has the statistics".format(m))
            with open(os.path.join(m, "cdspritesplus_stats.txt"), 'r') as stream:
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
            try:
                latest_ckpt = max(glob.glob(os.path.join(m, "model/" 'last.ckpt')), key=os.path.getmtime)
                print("Model: {}".format(latest_ckpt))
                exp = MultimodalVAEInfer(latest_ckpt)
                model = exp.get_wrapped_model()
                eval_single_model(model)
                if classifiers is None:
                    assert args.level != 0, "Please provide the --level argument based on the dataset level"
                    classifiers = get_all_classifiers(args.level)
                output, output_joint = eval_all(model, classifiers)
                output["joint"] = output_joint["joint"]
            except:
                pass
        if output is not None:
            text_image, image_text, joint = fill_cats(text_image, image_text, joint, output)
    output_dict = {"Text-Image Strict":{"value":stat.mean(text_image[0]), "stdev":stat.stdev(text_image[0])},
                   "Text-Image Features":{"value":stat.mean(text_image[1]), "stdev":stat.stdev(text_image[1])},
                   "Image-Text Strict":{"value":stat.mean(image_text[0]), "stdev":stat.stdev(image_text[0])},
                   "Image-Text Features":{"value":stat.mean(image_text[1]), "stdev":stat.stdev(image_text[1])},
                   "Image-Text Letters":{"value":stat.mean(image_text[2]), "stdev":stat.stdev(image_text[2])},
                   "Joint Strict":{"value":stat.mean(joint[0]), "stdev":stat.stdev(joint[0])},
                   "Joint Features":{"value":stat.mean(joint[1]), "stdev":stat.stdev(joint[1])}}
    print_save_stats(output_dict, parent_dir, "cdspritesplus", args.level)


if __name__ == "__main__":
    from eval.infer import MultimodalVAEInfer
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mpath", type=str, help="path to the .ckpt model file. Relative or absolute")
    parser.add_argument("-m", "--multieval", type=str, help="path to parent directory with multiple models")
    parser.add_argument("-l", "--level", type=int, default=0, help="for multieval option, if statistics for individual models are not yet made"),
    args = parser.parse_args()
    assert not (args.mpath and args.multieval), "You can only provide one of these arguments: mpath or mutlieval (not both)"
    if args.mpath:
        exp = MultimodalVAEInfer(args.mpath)
        model = exp.get_wrapped_model()
        eval_single_model(model)
    else:
        eval_cdsprites_over_seeds(args.multieval)
