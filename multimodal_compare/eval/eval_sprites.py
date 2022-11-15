# code for qualitative evaluation of multimodal VAEs trained on GeBiD dataset
import argparse
import glob, yaml
import numpy as np
import statistics as stat
from eval.sprites_classifier import VideoGPT
import torch
import os, glob
from utils import listdirs, last_letter, print_save_stats
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DIRECTIONS = ['front', 'left', 'right']
ACTIONS = ['walk', 'spellcard', 'slash']
LABEL_MAP = ["walk front", "walk left", "walk right", "spellcard front", "spellcard left",
                  "spellcard right", "slash front", "slash left", "slash right"]
ATTR_MAP = ["skin", "pants", "top", "hair"]
ATT_NAMES = [["pink", "yellow", "grey", "silver", "beige", "brown"],
                  ["white", "gold", "red", "armor", "blue", "green"],
                  ["maroon", "blue", "white", "armor", "brown", "shirt"],
                  ["green", "blue", "yellow", "silver", "red", "purple"]]


def action_labels(data):
    actions = np.argmax(data[:, :9], axis=-1)
    labels = []
    for a in actions:
        labels.append(LABEL_MAP[int(a)])
    return [labels]

def att_labels(data):
    atts = np.argmax(data, axis=-1)
    labels = []
    for a in atts:
        at = []
        for i, val in enumerate(a):
            str = ATT_NAMES[i][int(val)] + " " + ATTR_MAP[i]
            at.append(str)
        labels.append(at)
    l1 = [l[0] for l in labels]
    l2 = [l[1] for l in labels]
    l3 = [l[2] for l in labels]
    l4 = [l[3] for l in labels]
    return [l1, l2, l3, l4]

def keep_one_mod(data, mod_num:int):
    data["mod_{}".format(mod_num)]["data"] = data["mod_{}".format(mod_num)]["data"].cuda()
    for i in range(3):
        n = i+1
        if n != mod_num:
            data["mod_{}".format(n)] = {"data": None, "masks": None}
    return data


def load_classifier(class_type:str):
    model = VideoGPT(class_type).to("cuda")
    if class_type == "action":
        model.load_state_dict(torch.load("./data/sprites/sprites_classifier_frame2action.pth"))
    elif class_type == "attributes":
        model.load_state_dict(torch.load("./data/sprites/sprites_classifier_frame2attributes.pth"))
    return model

def calculate_cross_coherency(model_exp):
    """
    Calculates the cross-coherency accuracy for the given model (Frames -> Atts, Atts -> Frames, Frames -> Actions and
    Actions -> Frames)

    :param model: multimodal VAE
    :type model: object
    :return: mean cross accuracies
    :rtype: dict
    """
    acc_all = {"frames2actions": [0], "actions2frames": [0], "frames2atts":[0], "atts2frames":[0]}
    testset, _ = model_exp.datamod.get_num_samples(250, split="test")
    i1 = testset.copy()
    i1 = keep_one_mod(i1, 1)
    output1 = model_exp.model.forward(i1)
    out_actions = torch.argmax(output1.mods["mod_2"].decoder_dist.loc, dim=-1)
    out_atts = torch.argmax(output1.mods["mod_3"].decoder_dist.loc, dim=-1)
    correct_atts, correct_actions = 0, 0
    for i, x in enumerate(testset["mod_3"]["data"]):
        if all(torch.argmax(testset["mod_3"]["data"][i].cuda(), dim=-1) == out_atts[i]):
           correct_atts += 1
    for i, x in enumerate(testset["mod_2"]["data"]):
         if torch.argmax(testset["mod_2"]["data"][i].cuda(), dim=-1) == out_actions[i]:
             correct_actions += 1
    acc_all["frames2actions"] = correct_actions / len(out_actions)
    acc_all["frames2atts"] = correct_atts / len(out_atts)
    correct_frames = eval_with_classifier(testset, model_exp, "action", 2)
    acc_all["actions2frames"] = correct_frames / len(testset["mod_2"]["data"])
    correct_frames = eval_with_classifier(testset, model_exp, "attributes", 3)
    acc_all["atts2frames"] = correct_frames / len(testset["mod_3"]["data"])
    return acc_all

def eval_with_classifier(testset, model_exp, mod_name, mod_idx):
    i = testset.copy()
    i = keep_one_mod(i, mod_idx)
    output = model_exp.model.forward(i)
    out_frames = output.mods["mod_1"].decoder_dist.loc * 255
    clsf = load_classifier(mod_name)
    recognized = torch.argmax(clsf.forward(out_frames), dim=-1)
    gt = torch.argmax(i["mod_{}".format(mod_idx)]["data"], dim=-1)
    correct_frames = 0
    for i, a in enumerate(recognized):
        if len(a.unsqueeze(-1)) > 1:
            if all(a == gt[i]):
                correct_frames += 1
        else:
            if a == gt[i]:
                correct_frames += 1
    return correct_frames

def calculate_joint_coherency(model_exp):
    """
    Calculates the joint-coherency accuracy for the given model

    :param model: multimodal VAE
    :type model: object
    :return: mean joint accuracy
    :rtype: dict
    """
    recons = model_exp.save_joint_samples(num_samples=64, savedir=os.path.join(model_exp.config.mPath, "visuals/"))
    d1 = recons["mod_1"]
    d2 = recons["mod_2"]
    d3 = [x.replace("\n", "").split(" ")[0::2][:-1] for x in recons["mod_3"]]
    cl1 = load_classifier("action")
    cl2 = load_classifier("attributes")
    recognized_action = torch.argmax(cl1.forward(torch.tensor(d1).cuda()), dim=-1)
    recognized_atts = torch.argmax(cl2.forward(torch.tensor(d1).cuda()), dim=-1)
    action_frame = 0
    att_frame = 0
    for i, d in enumerate(d3):
        matching_atts = 0
        for idx, a in enumerate(recognized_atts[i]):
            value = ATT_NAMES[idx][a]
            if value == d[idx]:
                matching_atts += 1
        action_matching = LABEL_MAP[recognized_action[i]] == d2[i]
        if action_matching:
            action_frame += 1
        if matching_atts == 4:
            att_frame += 1
    output = {"joint_action_frame":action_frame/len(d3), "joint_att_frame": att_frame/len(d3)}
    return output

def labelled_tsne(model_exp):
    """
    Plots the T-SNE visualizations with the GeBiD text labels (one T-SNE per each feature)

    :param model: multimodal VAE
    :type model: object
    """
    testset, labels = model_exp.datamod.get_num_samples(250, split="test")
    att_lbls = att_labels(testset["mod_3"]["data"])
    act_lbls = action_labels(testset["mod_2"]["data"])
    for idx, lbls in enumerate([act_lbls, att_lbls]):
        for i, label in enumerate(lbls):
            model_exp.analyse_data(testset, label, path_name="eval_{}_{}".format(idx,i), savedir=os.path.join(model_exp.config.mPath, "visuals"))
    print("Saved labelled T-SNEs in {}".format(os.path.join(model_exp.config.mPath, "visuals")))


def eval_all(model_exp):
    print("- Making T-SNE for each feature")
    labelled_tsne(model_exp)
    print("- Calculating cross coherency")
    output_cross = calculate_cross_coherency(model_exp)
    print("- Calculating joint coherency")
    output_joint = calculate_joint_coherency(model_exp)
    return output_cross, output_joint

def eval_single_model(m_exp):
    print("\nCalculating automatic statistics")
    output_cross, output_joint = eval_all(m_exp)
    output_dict = {"Frames to Action Cross-Coherency":{"value":output_cross["frames2actions"], "stdev":None},
                   "Action to Frames Cross-Coherency":{"value":output_cross["actions2frames"], "stdev":None},
                   "Frames to Attributes Cross-Coherency":{"value":output_cross["frames2atts"], "stdev":None},
                   "Attributes to Frames Cross-Coherency":{"value":output_cross["atts2frames"], "stdev":None},
                   "Frames and Action Joint-Coherency":{"value":output_joint["joint_action_frame"], "stdev":None},
                   "Frames and Attributes Joint-Coherency":{"value":output_joint["joint_att_frame"], "stdev":None},
                   "Total Average Accuracy":{"value":(sum(list(output_cross.values())) + sum(list(output_joint.values())))/6, "stdev":None}
                   }
    print_save_stats(output_dict, m_exp.config.mPath, "sprites")
    return output_cross, output_joint

def fill_cats(output_dict):
    for key in output_dict.keys():
        vals = [x[0] for x in output_dict[key]]
        mean = stat.mean(vals)
        stdev = stat.stdev(vals)
        output_dict[key] = {"value":mean, "stdev":stdev}
    return output_dict

def eval_over_seeds(parent_dir):
    all_models = listdirs(parent_dir)
    all_models = sorted(all_models, key=last_letter)
    output_dict = {"Frames to Action Cross-Coherency": [],
                   "Action to Frames Cross-Coherency": [],
                   "Frames to Attributes Cross-Coherency": [],
                   "Attributes to Frames Cross-Coherency": [],
                   "Frames and Action Joint-Coherency": [],
                   "Frames and Attributes Joint-Coherency": [],
                   "Total Average Accuracy": []
                   }
    for m in all_models:
        if os.path.exists(os.path.join(m, "sprites_stats.txt")):
            print("Model: {} already has the statistics".format(m))
            with open(os.path.join(m, "sprites_stats.txt"), 'r') as stream:
                d = yaml.safe_load(stream)
                ls = ["Frames to Action Cross-Coherency", "Action to Frames Cross-Coherency", "Frames to Attributes Cross-Coherency", "Attributes to Frames Cross-Coherency",
                      "Frames and Action Joint-Coherency", "Frames and Attributes Joint-Coherency", "Total Average Accuracy"]
                output = {}
                for a in ls:
                     output[a] = []
                     if a in d.keys():
                        value = d[a] if not isinstance(d[a], str) else float(d[a].split(" (")[0])
                        output[a].append(value)
        else:
            latest_version = max(glob.glob(os.path.join(m, "lightning_logs/" '*/')), key=os.path.getmtime)
            latest_ckpt = max(glob.glob(os.path.join(latest_version, "checkpoints",'*')), key=os.path.getmtime)
            print("Model: {}".format(latest_ckpt))
            exp = MultimodalVAEInfer(latest_ckpt)
            model = exp.get_wrapped_model()
            output_c, output_joint = eval_all(model)
            output = output_c.copy()
            output.update(output_joint)
        for key in output_dict.keys():
            output_dict[key].append(output[key])
    output_dict = fill_cats(output_dict)
    print_save_stats(output_dict, parent_dir, "sprites")


if __name__ == "__main__":
    from eval.infer import MultimodalVAEInfer
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mpath", type=str, help="path to the .ckpt model file. Relative or absolute")
    parser.add_argument("-m", "--multieval", type=str, help="path to parent directory with multiple models")
    args = parser.parse_args()
    assert not (args.mpath and args.multieval), "You can only provide one of these arguments: mpath or mutlieval (not both)"
    if args.mpath:
        exp = MultimodalVAEInfer(args.mpath)
        model = exp.get_wrapped_model()
        eval_single_model(model)
    else:
        eval_over_seeds(args.multieval)
