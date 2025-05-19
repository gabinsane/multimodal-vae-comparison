
import gymnasium as gym
import lanro_gym
import pickle, math
import numpy as np
import cv2, os
import glob
import sys
import torch
import random
import copy
import matplotlib.pyplot as plt
import time, os
from pathlib import Path
from typing import List
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user
import pybullet as p
from eval.infer import MultimodalVAEInfer
from models.datasets import DRAWER

SLOW_VIS = False
SCENES = {"1":"move left", "2":"move right", "3":"lift", "4":["move right", "lift"], "5":["move left", "lift", "move right"], "6":"reach", "7":"lift",
          "a":1, "b":1, "c":2, "d":3}
OBJECTS = ["lemon", "apple", "soap"]
FIXEDPOSE = [0,0,0.05]
X_axis_pos = [0,0.24]
Y_axis_pos = [-0.12,0.12]

def default_rot():
    return list(env.env.env.env.robot.default_arm_orn_RPY)

def make_step(pose):
    if len(pose) == 4:
        pose = pose[:3] + list(env.env.env.env.robot.default_arm_orn_RPY) + [pose[-1]]
    env.step(pose)
    if render:
        #pass
        time.sleep(.1)

def get_quat(euler):
    return p.getQuaternionFromEuler(euler)

def robot_pose():
    return env.env.env.env.robot.get_ee_position()

def handle_pos():
    return env.env.env.env.sim.get_link_state("drawer", 1)[0]

def object_pos(id):
    return env.env.env.env.sim.get_base_position(id)

def drawer_pos():
    return env.env.env.env.sim.get_link_state("drawer", 2)[0]

def random__pos(x_var=True, y_var=True):
    x = random.uniform(*(X_axis_pos)) if x_var else 0
    y = random.uniform(*(Y_axis_pos)) if y_var else 0
    dr_pos = [x, y, 0.05]
    return dr_pos

def random__pos_when_drawer(x_var=True, y_var=True):
    x = random.uniform(*([-0.1,0])) if x_var else 0
    y = random.choice([random.uniform(*([-0.19,-0.16])), random.uniform(*([0.16,0.19]))]) if y_var else 0.18
    dr_pos = [x, y, 0.05]
    return dr_pos

def random__pos_drawer(x_var=True, y_var=True):
    x = random.uniform(*([0.25,0.29])) if x_var else 0.25
    y = random.uniform(*([-0.04,0.04])) if y_var else 0
    dr_pos = [x, y, 0.05]
    return dr_pos

def get_handle_moving_point(opening=True):
    handle = list(handle_pos())
    x1, y1 = handle[0], handle[1]
    drawer = list(drawer_pos())
    x0, y0 = drawer[0], drawer[1]
    dx = x1 - x0
    dy = y1 - y0
    t = 0.16 if opening else -0.16
    length = ((dx)**2 + (dy)**2)**0.5
    
    scale_factor = t / length

    x2 = x1 + scale_factor * dx
    y2 = y1 + scale_factor * dy
    z2 = handle[2]
    env.env.env.env.sim.bclient.addUserDebugText("x", [x2, y2, z2], textSize=1.0, replaceItemUniqueId=3)
    return [x2, y2, z2]


def goto_pose(pose):
    r_p = robot_pose()
    counter = 0
    while sum(abs(robot_pose() -  pose[:3])) > 0.01:
        counter += 1
        make_step(pose)
        if sum(abs(robot_pose() - r_p)) < 0.001: 
            break # robot is no longer moving, probably physical constraints
        else:
            r_p = robot_pose()
        if counter == 20:
            print("moving glitch")
            break
        if SLOW_VIS:
            time.sleep(.1)

def go_above(pose):
    above_pose = pose
    above_pose[2] = 0.25
    goto_pose(above_pose)

def set_gripper(current_pose, open:bool, perform_step=False, rotation=None):
    g = 1 if open else -1
    current_pose = list(current_pose)
    if len(current_pose) in [4,8]:
        if rotation is None:
            current_pose[-1] = g
        else:
            current_pose = current_pose[:3] + rotation + [g]
    elif len(current_pose) == 3:
        if rotation is not None:
            current_pose += rotation
        current_pose.append(g)
    else:
        raise Exception("Wrong action size {}".format(len(current_pose)))
    if perform_step:
        make_step(current_pose)
    return current_pose

def spawn_drawer():
    drawer_id = env.env.env.env.sim.loadURDF(body_name="drawer",
                                fileName="./models/lanro_gym/objects_urdfs/cabinet.urdf", useFixedBase=True)
    rand_orn = random.uniform(-0.2,0.2)
    rand_pos = [random.uniform(0.14,0.35), random.uniform(-0.2,0.2), 0.1]
    env.env.env.env.sim.set_base_pose("drawer", rand_pos, [0, 0, rand_orn, 1])
    p.changeVisualShape(drawer_id, 0, rgbaColor=[0.5, 0.3, 0.1, 1])
    p.changeVisualShape(drawer_id, -1, rgbaColor=[0.8, 0.5, 0.3, 1])

def setup_scene(objects, targetobject, env, drawer=False):
    poses = [FIXEDPOSE] if (task[-1] == "a" and int(task[1]) <7) else [random__pos() for _ in objects]
    if int(task[1]) == 7:
        if task[2] in ["a", "b"]:
            poses = [random__pos(y_var=False)]
        else:
            poses = [random__pos_when_drawer(y_var=False)]
    if int(task[1]) > 7:
        if task[2] in ["a", "b"]:
            poses = [random__pos()]
        else:
            poses = [random__pos_when_drawer()]   
    targetobj_id, drawer_id = None, None
    o_ids = []
    for i, o in enumerate(objects):
        oid = env.env.env.env.sim.loadURDF(body_name="object_{}".format(o),
                                    fileName="./models/lanro_gym/objects_urdfs/{}.urdf".format(o), useFixedBase=False)
        if o == targetobject:
            targetobj_id = "object_{}".format(o)
        o_ids.append("object_{}".format(o))
        env.env.env.env.sim.set_base_pose("object_{}".format(o), poses[i], [0, 0, 0, 1])
    if drawer:
        drawer_id = env.env.env.env.sim.loadURDF(body_name="drawer",
                                    fileName="./models/lanro_gym/objects_urdfs/cabinet.urdf", useFixedBase=True)
        p.changeVisualShape(drawer_id, 0, rgbaColor=[0.5, 0.3, 0.1, 1])
        p.changeVisualShape(drawer_id, -1, rgbaColor=[0.8, 0.5, 0.3, 1])
        pos = random__pos_drawer(y_var=False) if int(task[1]) == 7 else random__pos_drawer()
        env.env.env.env.sim.set_base_pose("drawer", [pos[0], pos[1], 0.1], [0, 0, 0, 1])
        o_ids.append("drawer")
        manual_drawer_open(drawer_id)
    return o_ids, targetobj_id, drawer_id


def rotate_gripper_to_drawer(perform=True):
    gripper_rot=default_rot()
    last_rot = env.env.env.env.robot.sim.get_link_state(env.env.env.env.robot.body_name,6)[1]
    gripper_rot[1] = env.env.env.env.sim.get_base_orientation("drawer")[2]
    rpose = robot_pose()
    rpose[-1] = 0.2
    counter = 0
    while True:
        set_gripper(rpose, open=True, rotation=gripper_rot, perform_step=perform)
        counter += 1
        rot = env.env.env.env.robot.sim.get_link_state(env.env.env.env.robot.body_name,6)[1]
        if (sum(abs(np.array(rot) - np.array(last_rot))) < 0.001) or perform== False:
            break
        last_rot = rot
        if counter == 20:
            print("rotating glitch")
            break
    return gripper_rot


def manual_drawer_open(id):
    p.setJointMotorControlArray(bodyUniqueId=id,
            jointIndices=[0],
            controlMode=env.env.env.sim.bclient.POSITION_CONTROL,
            targetPositions=[2.2],
            forces=[10],)

def manual_drawer_closed(id):
    p.setJointMotorControlArray(bodyUniqueId=id,
            jointIndices=[0],
            controlMode=env.env.env.sim.bclient.POSITION_CONTROL,
            targetPositions=[0],
            forces=[100],)

def get_mod_idx(name):
    for key, item in infer_model.model.mod_names.items():
        if item == name:
            return key

def reach(targetobj_id):
    object_pose = set_gripper(object_pos(targetobj_id), open=True)
    go_above(object_pose)
    obj_pos = list(object_pos(targetobj_id))
    goto_pose(set_gripper(obj_pos, open=True))

def get_img(env):
    return env.env.env.env.robot.get_camera_img()

def process_model_output(output, mod_name):
    return output.mods[get_mod_idx(mod_name)].decoder_dist.loc.detach().cpu().numpy()


def cast_dict_to_cuda(d):
    for key in d.keys():
        if d[key]["data"] is not None:
            d[key]["data"] = d[key]["data"].cuda()
        if d[key]["masks"] is not None:
            d[key]["masks"] = d[key]["masks"].cuda()
    return d

def check_posdiff_enough(instruction, init_pos, final_pos):
    correct = False
    zdiff = abs(final_pos[-1] - init_pos[-1])
    distance = None
    if "right" in instruction:
        if final_pos[1] - init_pos[1] > 0.2 and zdiff < 0.1:
            correct = True
    elif "left" in instruction:
        if init_pos[1] - final_pos[1] > 0.2 and zdiff < 0.1:
            correct = True
    elif "lift" in instruction:
        if zdiff > 0.07:
            correct = True
    elif "reach" in instruction:
        distance = sum(abs(np.array(final_pos) - np.array((robot_pose())[:3])))
        if sum(abs(np.array(final_pos) - np.array((robot_pose())[:3]))) < 0.06:
            correct = True
    elif instruction == "put in":
        if sum(abs(np.array(final_pos)-np.array(drawer_pos()))) < 0.19:
            correct = True
    elif instruction == "close drawer":
        if sum(abs(np.array(final_pos)-np.array(drawer_pos()))) < 0.19 and sum(abs(np.array(init_pos)-np.array(handle_pos()))) > 0.1:
            correct = True
    return correct, distance

def empty_steps(num):
    for x in range(num):
         env.step(list(robot_pose()) +  list(env.env.env.env.robot.default_arm_orn_RPY) + [1])

def get_instruction_action(setup):
    if setup[2] == "a":
        action_in = instruction = "reach"
    elif setup[2] == "b":
        action_in = instruction = "lift"
    elif setup[2] == "c":
        action_in = instruction = "put in"
    else:
        action_in = instruction = "close drawer"     
    return action_in, instruction

def infer_loop(setup, infer_model):
        global robot_pos
        global robot_joint
        robot_pos = []
        robot_joint = []
        drawer_pos = None
        env.reset()
        if setup[1] == "9":
            env.env.env.env.sim.set_base_pose("panda", [-0.6,random.uniform(-0.2,0.2),0], (0.0, 0.0, 0.0, 1.0))
        objects = random.sample(OBJECTS, SCENES[setup[-1]])
        if (int(setup[1]) > 6 and setup[2] in ["c","d"]):
            objects = [objects[0]]
        targetobject = random.choice(objects)
        drawer = True if (int(setup[1]) > 6 and setup[2] in ["c","d"]) else False
        o_ids, targetobj_id, drawer_id = setup_scene(objects, targetobject, env, drawer)
        empty_steps(5)
        if drawer:
            init_handle_pos = handle_pos()
        image = env.env.env.env.robot.get_camera_img()
        image = cv2.resize(image, (64,64))
        if int(setup[1]) < 7:
            action_in = SCENES[setup[1]]
            if isinstance(action_in, list):
                action_in = random.choice(action_in)
            obj_in = " ".join(["the", targetobject]) if setup[-1] not in ["a", "b"] else ""
            instruction = " ".join([action_in, obj_in]).strip()
        else:
                action_in, instruction = get_instruction_action(setup)
        #env.env.env.env.sim.bclient.addUserDebugText(setup + " " + instruction, [0.15, -.3, .6], textSize=2.0, replaceItemUniqueId=43)
        init_obj_pos = object_pos(targetobj_id)


        mods_batch = {}
        if int(setup[1]) in [4,5] or (int(setup[1])<7 and setup[2] in ["c", "d"]):
            mods_batch["mod_1"] = {"data": preprocess_instruction(instruction), "masks":None}
            mods_batch["mod_2"] = {"data": None, "masks":None}
            mods_batch["mod_3"] = {"data": torch.tensor(image/255).reshape(3,64,64).unsqueeze(0).float(), "masks":None}
        else:
            mods_batch["mod_1"] = {"data": None, "masks":None}
            mods_batch["mod_2"] = {"data": torch.tensor(image/255).reshape(3,64,64).unsqueeze(0).float(), "masks":None}
        output = infer_model.model.model.forward(cast_dict_to_cuda(mods_batch))
        actions = process_model_output(output, "actions")
        min_distance = None
        for i, pose in enumerate(actions[0][:70]):
                gripper = 1
                if pose[-1] < -0.5:
                    gripper = -1
                make_step([float(p) for p in pose[:-1]] + [gripper])
                if drawer:
                    init_obj_pos = init_handle_pos
                correct, distance = check_posdiff_enough(instruction, init_obj_pos, object_pos(targetobj_id))
                if min_distance is None and "reach" in instruction:
                    min_distance = distance
                elif "reach" in instruction and distance < min_distance:
                    min_distance = distance
                if correct:
                    if "drawer" in instruction:
                        manual_drawer_closed(drawer_id)
                    break
        if "put" in instruction:
               empty_steps(15)
               correct, dist = check_posdiff_enough(instruction, init_obj_pos, object_pos(targetobj_id))
        for x in o_ids:
            env.env.env.env.sim.remove_body(x)
        if not correct:
            return False, min_distance
        else:
            print("Correct")
            return True, min_distance

def load_vocab(pth):
    vocab = []
    with open(pth, "r") as f:
        for line in f:
                vocab.append(line.replace("\n", ""))
    return vocab

def preprocess_instruction(instruction):
    indices = [vocab.index(s) for s in instruction.split(" ")]
    data = torch.nn.functional.one_hot(torch.tensor(np.array(indices)), num_classes=9)
    return data.unsqueeze(0)

if __name__ == '__main__':
    model_ckpts = glob.glob("")
    for model_ckpt in model_ckpts:
        print(model_ckpt)
        if not os.path.exists(os.path.join(os.path.dirname(model_ckpt), 'success_percentage.txt')):
            model_cfg = (os.path.dirname(os.path.dirname(model_ckpt)) + "/config.yml")
            task = model_ckpt.split("/")[-4].split("_")[0]
            render = True
            infer_model = MultimodalVAEInfer(model_ckpt, model_cfg)
            success = []
            min_distances = []
            env = gym.make("PandaEmpty-v0", render=render)
            vocab = load_vocab("./data/lanro_long/D6a/vocab.txt")
            for x in range(500):
                if x % 20 == 0:
                    print(x)
                    env.close()
                    env = gym.make("PandaEmpty-v0", render=render)
                env.reset()
                succ, min_distance = infer_loop(task, infer_model)
                min_distances.append(min_distance)
                if succ:
                    success.append(1)
                else:
                    success.append(0)
            succ = (sum(success)/len(success))*100
            print(model_ckpt)
            print("Success percentage: {}".format(succ))
            with open(os.path.join(os.path.dirname(model_ckpt), 'success_percentage.txt'), 'w') as f:
                 f.write(str(succ))
            env.close()
