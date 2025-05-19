"""A script to demonstrate grasping."""
import gymnasium as gym
import lanro_gym
import time as time
import numpy as np
import random
import pickle
import pybullet as p
import math
import sys, os
import cv2

SLOW_VIS = False
SCENES = {"1":"move left", "2":"move right", "3":"lift", "4":["move left", "lift"], "5":["move left", "lift", "move right"], "6":"reach", "7":"reach",
          "a":1, "b":1, "c":2, "d":3}
OBJECTS = ["lemon", "apple", "soap"]
FIXEDPOSE = [0,0,0.05]
X_axis_pos = [0,0.24]
Y_axis_pos = [-0.12,0.12]


def calculate_angle_2d(pos1, pos2):
    # Calculate the angle in radians using atan2
    x0, y0 = pos1[:2]
    x1,y1 = pos2[:2]
    angle_rad = math.atan2(y1 - y0, x1 - x0)
    # Convert the angle to degrees
    #angle_deg = math.degrees(angle_rad)
    return angle_rad

def default_rot():
    return list(env.env.env.env.robot.default_arm_orn_RPY)

def make_step(pose):
    if len(pose) == 4:
        pose = pose[:3] + list(env.env.env.env.robot.default_arm_orn_RPY) + [pose[-1]]
    env.step(pose)
    robot_joint.append([*env.env.env.env.robot.get_current_pos(), pose[-1]])
    robot_pos.append(list(robot_pose()) + [pose[-1]])
    if render:
        pass
        #time.sleep(.1)

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
    poses = [FIXEDPOSE] if (setup[-1] == "a" and int(setup[1]) <7) else [random__pos() for _ in objects]
    if int(setup[1]) == 7:
        if setup[2] in ["a", "b"]:
            poses = [random__pos(y_var=False)]
        else:
            poses = [random__pos_when_drawer(y_var=False)]
    if int(setup[1]) > 7:
        if setup[2] in ["a", "b"]:
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
        pos = random__pos_drawer(y_var=False) if int(setup[1]) == 7 else random__pos_drawer()
        env.env.env.env.sim.set_base_pose("drawer", [pos[0], pos[1], 0.1], [0, 0, 0, 1])
        o_ids.append("drawer")
        manual_drawer_open(drawer_id)
    return o_ids, targetobj_id, drawer_id

def add_to_list():
    images.append(image)
    #cv2.imwrite("exampleimage.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    instructions.append(instruction)
    obj_poses.append(list(obj_pos))
    robot_poses.append(robot_pos)
    object_colors.append(col_lib)
    object_shapes.append(shape_lib)
    correct_objects.append(correct_object)
    robot_actions.append(robot_action)
    robot_ees.append(robot_ee)

def grasp_handle():
    g_rot = rotate_gripper_to_drawer()
    handle_position = handle_pos()
    above_handle = list(handle_position)
    above_handle = set_gripper(above_handle, rotation=g_rot, open=True)
    goto_pose(above_handle)
    hold_handle = set_gripper(above_handle, open=False, rotation=g_rot, perform_step=True)
    return hold_handle

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

def open_close_drawer(open=True, drawerid=None):
    beginning_handle_pose = handle_pos()
    g_rot = rotate_gripper_to_drawer(perform=False)
    grasp_handle()
    pull_handle = get_handle_moving_point(opening=open)
    pull_handle = set_gripper(pull_handle, open=False, rotation=g_rot, perform_step=True)
    goto_pose(pull_handle)
    success = True
    if open == False:
        success = False
        if open == False and sum(abs(np.array(beginning_handle_pose)-np.array(handle_pos()))) > 0.10:
            manual_drawer_closed(drawerid)
            success = True
    let_go = set_gripper(pull_handle, open=True, perform_step=True)
    go_above(let_go)
    return success

def put_in(id):
    object_pose = set_gripper(object_pos(id), open=True)
    go_above(object_pose)
    goto_pose(set_gripper(object_pos(id), open=True))
    set_gripper(object_pos(id), open=False, perform_step=True)
    go_above(set_gripper(object_pos(id), open=False))
    drawer_pose = set_gripper(drawer_pos(), open=False)
    drawer_pose[0] += 0.02
    go_above(drawer_pose)
    let_go = set_gripper(drawer_pose, open=True, perform_step=True)
    go_above(let_go)

def manual_drawer_closed(id):
    p.setJointMotorControlArray(bodyUniqueId=id,
            jointIndices=[0],
            controlMode=env.env.env.sim.bclient.POSITION_CONTROL,
            targetPositions=[0],
            forces=[100],)

def take_out():
    object_pose = set_gripper(object_pos(), open=True)
    go_above(object_pose)
    obj_pos = list(object_pos())
    goto_pose(set_gripper(obj_pos, open=True))
    set_gripper(obj_pos, open=False, perform_step=True)
    go_above(set_gripper(obj_pos, open=False))
    outside_pose = set_gripper(random_outside_pos(), open=False)
    go_above(outside_pose)
    let_go = set_gripper(outside_pose, open=True, perform_step=True)
    for x in range(6):
        set_gripper(let_go, open=True, perform_step=True)

def grasp(targetobj_id):
    object_pose = set_gripper(object_pos(targetobj_id), open=True)
    go_above(object_pose)
    obj_pos = list(object_pos(targetobj_id))
    goto_pose(set_gripper(obj_pos, open=True))
    set_gripper(obj_pos, open=False, perform_step=True)

def reach(targetobj_id):
    object_pose = set_gripper(object_pos(targetobj_id), open=True)
    go_above(object_pose)
    obj_pos = list(object_pos(targetobj_id))
    goto_pose(set_gripper(obj_pos, open=True))

def get_img(env):
    return env.env.env.env.robot.get_camera_img()

def newpose_from_instruction(instruction):
    current_pose = list(robot_pose())
    current_pose[2] += 0.05  
    if "left" in instruction:
        current_pose[1] -= 0.4
    elif "right" in instruction:
        current_pose[1] += 0.4
    elif "lift" in instruction:
        current_pose[2] += 0.2
    
    return current_pose  

def check_posdiff_enough(instruction, init_pos, final_pos):
    correct = False
    zdiff = abs(final_pos[-1] - init_pos[-1])
    if "right" in instruction:
        if final_pos[1] - init_pos[1] > 0.2 and zdiff < 0.1:
            correct = True
    elif "left" in instruction:
        if init_pos[1] - final_pos[1] > 0.2 and zdiff < 0.1:
            correct = True
    elif "lift" in instruction:
        if zdiff > 0.1:
            correct = True
    elif "reach" in instruction:
        if abs(sum(np.array(final_pos) - np.array((robot_pose())[:3]))) < 0.06:
            correct = True
    elif instruction == "put in":
        if sum(abs(np.array(final_pos)-np.array(drawer_pos()))) < 0.16:
            correct = True
    elif instruction == "close drawer":
        if sum(abs(np.array(final_pos)-np.array(drawer_pos()))) < 0.16 and sum(abs(np.array(drawer_pos())-np.array(handle_pos()))) < 0.18:
            correct = True
    return correct

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

def data_loop(env, setup):
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
        image = env.env.env.env.robot.get_camera_img()
        #image = cv2.resize(image, (64,64))
        if int(setup[1]) < 7:
            action_in = SCENES[setup[1]]
            if isinstance(action_in, list):
                action_in = random.choice(action_in)
            obj_in = " ".join(["the", targetobject]) if setup[-1] not in ["a", "b"] else ""
            instruction = " ".join([action_in, obj_in]).strip()
        else:
                action_in, instruction = get_instruction_action(setup)
        env.env.env.env.sim.bclient.addUserDebugText(instruction, [0.15, -.3, .6], textSize=2.0, replaceItemUniqueId=43)
        init_obj_pos = object_pos(targetobj_id)
        success = True
        if action_in != "reach":
            if action_in in ["put in", "close drawer"]:
                put_in(targetobj_id)
                empty_steps(50)
                if action_in == "close drawer":
                    success = open_close_drawer(open=False, drawerid=drawer_id)
            else:
                grasp(targetobj_id)
                goto_pose(set_gripper(newpose_from_instruction(instruction), open=False))
        else:
            reach(targetobj_id)
        correct = check_posdiff_enough(instruction, init_obj_pos, object_pos(targetobj_id))
        for x in o_ids:
            env.env.env.env.sim.remove_body(x)
        if not correct or not success:
            print("FAIL")   
            return [None, None, None, None, None, None]       
        else:
            return [robot_pos, robot_joint, instruction, list(init_obj_pos), image, drawer_pos]

def manual_drawer_open(id):
    p.setJointMotorControlArray(bodyUniqueId=id,
            jointIndices=[0],
            controlMode=env.env.env.sim.bclient.POSITION_CONTROL,
            targetPositions=[2.2],
            forces=[10],)

def make_dataset(episodes=100, folder="unknown", render=False):
    kwargs = {"render":render}
    for runx in range(0,episodes):
        obj_poses = []
        instructions = []
        robot_joints = []
        drawer_positions = []
        robot_actions = []
        images = []
        os.makedirs(folder, exist_ok=True)
        collector = [robot_actions, robot_joints, instructions, obj_poses, images, drawer_positions]
        global env
        env = gym.make("PandaEmpty-v0", **kwargs)
        while not len(instructions) == batchsize:
            print(len(instructions))
            outputs = data_loop(env, folder)
            for i, o in enumerate(outputs):
                if o is not None:
                    if i == 4:
                        cv2.imwrite("./{}/image{}_{}.png".format(folder, runx, len(instructions)), cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
                    collector[i].append(o)
        env.close()
        with open(os.path.join(folder, 'robot_joints_{}.pkl'.format(runx)), 'wb') as handle:
                pickle.dump(robot_joints, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(folder, 'endeff_actions_{}.pkl'.format(runx)), 'wb') as handle:
            pickle.dump(robot_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if len(drawer_positions) > 0:
            with open(os.path.join(folder, 'drawer_poses_{}.pkl'.format(runx)), 'wb') as handle:
                pickle.dump(drawer_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(folder, 'instructions_{}.pkl'.format(runx)), 'wb') as handle:
            pickle.dump(instructions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(folder, 'objpose_{}.pkl'.format(runx)), 'wb') as handle:
            pickle.dump(obj_poses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(folder, 'image_{}.pkl'.format(runx)), 'wb') as handle:
            pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__== "__main__":
    setups = ["D7d"]
    for setup in setups:
        size = 10000
        batchsize = 50
        render = True
        make_dataset(episodes=int(size/batchsize), folder=setup, render=render)
