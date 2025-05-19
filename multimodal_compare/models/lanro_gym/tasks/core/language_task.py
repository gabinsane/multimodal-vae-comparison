import itertools
from typing import Dict, List, Tuple

import numpy as np
from lanro_gym.robots.pybrobot import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.scene import basic_scene
from lanro_gym.env_utils import RGBCOLORS, TaskObjectList, SHAPES, WEIGHTS, SIZES, valid_task_object_combination, distinguishable_by_primary_or_secondary
from lanro_gym.language_utils import create_commands, word_in_string
import itertools


class LanguageTask:
    distance_threshold: float = 0.05
    object_size: float = 0.04
    current_instruction: np.ndarray
    action_verbs: List = []
    secondary_props_words: List = []
    obj_range_low: np.ndarray
    obj_range_high: np.ndarray
    task_object_list: TaskObjectList
    num_obj: int = 0
    np_random: np.random.Generator

    # hindsight instruction flags and metrics
    hindsight_instruction = None
    use_hindsight_instructions: bool = False
    total_hindsight_instruction_episodes: int = 0
    discovered_hindsight_instruction_ctr: int = 0

    # action repair flags and metrics
    use_action_repair: bool = False
    use_negations_action_repair: bool = False
    ep_action_repair_returned: bool = False
    action_repair_success_ctr: int = 0
    action_repair_ctr: int = 0
    total_action_repair_episodes: int = 0

    # pybullet user debug text ids
    instruction_sim_id = 43
    action_repair_sim_id = 44

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 mode: str,
                 use_hindsight_instructions: bool,
                 use_action_repair: bool,
                 delay_action_repair: bool,
                 use_negations_action_repair: bool,
                 num_obj: int,
                 use_synonyms: bool = False):
        self.sim = sim
        self.robot = robot
        self.use_hindsight_instructions = use_hindsight_instructions
        self.use_action_repair = use_action_repair
        self.use_negations_action_repair = use_negations_action_repair
        self.num_obj = num_obj
        self.mode = mode
        self.use_synonyms = use_synonyms

        self.delay_action_repair = delay_action_repair
        self.ep_delayed_ar_command = None

        _args = dict(
            color_mode='color' in mode,
            shape_mode='shape' in mode,
            size_mode='size' in mode,
            weight_mode='weight' in mode,
        )
        self.task_object_list = TaskObjectList(sim, **_args)
        self.object_properties = self.task_object_list.get_obj_properties()

    def get_goal(self) -> str:
        """ Get the goal as instruction string """
        return self.current_instruction.item()

    def reset(self):
        """Reset the task: sample a new goal"""
        raise NotImplementedError

    def is_success(self) -> float:
        raise NotImplementedError

    def compute_reward(self) -> float:
        raise NotImplementedError

    def _create_scene(self) -> None:
        basic_scene(self.sim)

    def get_all_instructions(self) -> List[str]:
        instruction_set = np.concatenate([
            create_commands("instruction",
                            _property_tuple,
                            action_verbs=self.action_verbs,
                            use_synonyms=self.use_synonyms) for _property_tuple in self.object_properties
        ])
        if self.use_action_repair:
            action_repair_set = np.concatenate([
                create_commands("repair", _property_tuple, use_synonyms=self.use_synonyms)
                for _property_tuple in self.object_properties
            ])
            negations = np.concatenate([
                create_commands("negation", _property_tuple, use_synonyms=self.use_synonyms)
                for _property_tuple in self.object_properties
            ]) if self.use_negations_action_repair else []
            action_repair_set = np.concatenate([action_repair_set, negations])
            # combine each instruction with each action repair command
            # NOTE: concatenating two huge arrays will make the SWAP MEM explode
            # -> For-Loop is slow but works
            instruction_action_repair_set = [
                _instr + ' ' + _ar for _instr, _ar in itertools.product(instruction_set, action_repair_set)
            ]
            instruction_set = np.concatenate((instruction_set, instruction_action_repair_set))
        return list(set(instruction_set))

    def get_instructions_by_properties(self):
        inst_properties = {}
        inst_set = self.get_all_instructions()
        for inst in inst_set:
            if inst not in inst_properties.keys():
                inst_properties[inst] = {"color": '', "shape": '', "weight": '', "size": ''}
            for obj_prop in np.concatenate([RGBCOLORS, SHAPES, WEIGHTS, SIZES]):
                if isinstance(obj_prop, RGBCOLORS):
                    inst_word = word_in_string(inst, [obj_prop.name.lower()])
                    if inst_word:
                        inst_properties[inst]["color"] = inst_word
                if isinstance(obj_prop, SHAPES):
                    inst_word = word_in_string(inst, obj_prop.value[1])
                    if inst_word:
                        inst_properties[inst]["shape"] = inst_word
                if isinstance(obj_prop, WEIGHTS):
                    inst_word = word_in_string(inst, obj_prop.value[1])
                    if inst_word:
                        inst_properties[inst]["weight"] = inst_word
                if isinstance(obj_prop, SIZES):
                    inst_word = word_in_string(inst, obj_prop.value[1])
                    if inst_word:
                        inst_properties[inst]["size"] = inst_word
        return inst_properties

    def get_obs(self) -> np.ndarray:
        observation = []
        for idx in self.obj_indices_selection:
            obj_key = f"object{idx}"
            object_position = np.array(self.sim.get_base_position(obj_key))
            object_rotation = np.array(self.sim.get_base_rotation(obj_key))
            object_velocity = np.array(self.sim.get_base_velocity(obj_key))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity(obj_key))
            object_identifier = self.task_object_list.objects[idx].get_onehot()
            observation.extend(
                [object_position, object_rotation, object_velocity, object_angular_velocity, object_identifier])
        return np.concatenate(observation)

    def get_contact_with_fingers(self, target_body) -> List:
        # check contact with fingers defined by ee_joints
        finger_contacts = [
            bool(self.sim.get_contact_points(target_body, self.robot.body_name, linkIndexB=finger_idx))
            # assume the first two indices are the fingers of the end effector
            for finger_idx in self.robot.ee_joints[:2]
        ]
        return finger_contacts

    def _sample_goal(self) -> None:
        """Randomly select one of the generated instructions for the current goal object"""
        property_tuple = self.task_object_list.objects[self.goal_obj_idx].get_properties()
        sentences = create_commands("instruction",
                                    property_tuple,
                                    action_verbs=self.action_verbs,
                                    use_synonyms=self.use_synonyms)
        self.current_instruction = self.np_random.choice(sentences, 1)
        # self.sim.bclient.addUserDebugText(self.get_goal(), [0.05, -.3, .4],
        #                                   textSize=2.0,
        #                                   replaceItemUniqueId=self.instruction_sim_id)

    def _sample_objects(self) -> Tuple:
        """Randomize start position of objects."""
        while True:
            obj_positions = [[0.0, 0.0, self.object_size / 2] +
                             self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                             for _ in range(self.num_obj)]
            unique_distance_combinations = [np.linalg.norm(a - b) for a, b in itertools.combinations(obj_positions, 2)]
            # if minimal distance between two objects is greater than three times
            # the object size (as objects should not be on top of each other
            # and we like to have a minimal distance between them)
            if np.min(unique_distance_combinations) > self.object_size * 3:
                return tuple(obj_positions)

    def is_unique_obj_selection(self, obj_list):
        for obj_tuple in itertools.combinations(obj_list, 2):
            obj1, obj2 = obj_tuple
            if not distinguishable_by_primary_or_secondary(obj1, obj2):
                return False
        return True

    def sample_task_objects(self, testing=False):
        # remove old objects, as we do not want to destroy the whole simulation
        remove_obj_keys = [key for key in self.sim._bodies_idx.keys() if 'object' in key]
        for _key in remove_obj_keys:
            self.sim.remove_body(_key)

        # Ensure we only have duplicates along one feature dimension
        while True:
            self.obj_indices_selection = self.np_random.choice(len(self.task_object_list),
                                                               size=self.num_obj,
                                                               replace=False)
            self.goal_obj_idx = self.np_random.choice(self.obj_indices_selection, 1)[0]
            self.non_goal_body_indices = [idx for idx in self.obj_indices_selection if idx != self.goal_obj_idx]
            valid_combination = np.all([
                valid_task_object_combination(self.task_object_list[self.goal_obj_idx],
                                              self.task_object_list[non_goal_idx])
                for non_goal_idx in self.non_goal_body_indices
            ])
            if self.use_action_repair:
                unique_obj_selection = self.is_unique_obj_selection(
                    [self.task_object_list[self.goal_obj_idx]] +
                    [self.task_object_list[non_goal_idx] for non_goal_idx in self.non_goal_body_indices])
                if valid_combination and unique_obj_selection:
                    break
            else:
                if valid_combination:
                    break

        self.goal_object_body_key = f"object{self.goal_obj_idx}"

        for obj_idx in self.obj_indices_selection:
            object_body_key = f"object{obj_idx}"
            if not testing:
                self.task_object_list.objects[obj_idx].load(object_body_key)
            else:
                self.task_object_list.objects[obj_idx].load_test(object_body_key)

    def return_delayed_action_repair(self):
        if self.ep_action_repair and self.delay_action_repair and self.ep_delayed_ar_command is not None:
            if self._delay_ctr == 0:
                self.action_repair_ctr += 1
                action_repair_command = self.ep_delayed_ar_command
                self.sim.bclient.addUserDebugText(self.get_goal(), [0.05, -.3, .4],
                                                  textSize=2.0,
                                                  replaceItemUniqueId=self.instruction_sim_id)
                self.sim.bclient.addUserDebugText(action_repair_command, [0.05, -.3, .35],
                                                  textSize=2.0,
                                                  textColorRGB=RGBCOLORS.ORANGE.value[0],
                                                  replaceItemUniqueId=self.action_repair_sim_id)
                self.current_instruction = np.array([self.get_goal() + ' ' + action_repair_command])
                self.ep_delayed_ar_command = None
            else:
                self._delay_ctr -= 1

    def merge_instruction_action_repair(self, repair_commands):
        action_repair_command = self.np_random.choice(repair_commands, 1)[0]
        if self.delay_action_repair and self.ep_delayed_ar_command is None:
            self.ep_delayed_ar_command = action_repair_command
            self._delay_ctr = self.np_random.integers(0, 20)
        else:
            self.action_repair_ctr += 1
            self.sim.bclient.addUserDebugText(self.get_goal(), [0.05, -.3, .4],
                                              textSize=2.0,
                                              replaceItemUniqueId=self.instruction_sim_id)
            self.sim.bclient.addUserDebugText(action_repair_command, [0.05, -.3, .35],
                                              textSize=2.0,
                                              textColorRGB=RGBCOLORS.ORANGE.value[0],
                                              replaceItemUniqueId=self.action_repair_sim_id)
            self.current_instruction = np.array([self.get_goal() + ' ' + action_repair_command])
            # we only want one action repair command per episode

        self.ep_action_repair_returned = True

    def generate_action_repair_or_success(self) -> float:
        if self.ep_action_repair and not self.ep_action_repair_returned:
            # Simulate false instruction:
            # randomly select another object as new goal
            self.goal_obj_idx = self.np_random.choice(self.non_goal_body_indices, 1)[0]
            self.non_goal_body_indices = [idx for idx in self.obj_indices_selection if idx != self.goal_obj_idx]
            self.goal_object_body_key = f"object{self.goal_obj_idx}"
            target_property_tuple = self.task_object_list.objects[self.goal_obj_idx].get_properties()
            # generate action repair command considering new goal
            repair_commands = create_commands("repair", target_property_tuple, use_synonyms=self.use_synonyms)
            self.merge_instruction_action_repair(repair_commands)
            return -1.0
        else:
            # if action repair already triggered and now successful with new goal (only count onces per episode)
            if self.ep_action_repair_returned and not self.ep_action_repair_success:
                # AR of this episode successful
                self.action_repair_success_ctr += 1
                self.ep_action_repair_success = True
            return 0.0

    def generate_hindsight_instruction(self, _obj_idx):
        property_tuple = self.task_object_list.objects[_obj_idx].get_properties()
        hindsight_sentences = create_commands("instruction",
                                              property_tuple,
                                              action_verbs=self.action_verbs,
                                              use_synonyms=self.use_synonyms)
        self.discovered_hindsight_instruction_ctr += 1
        self.ep_hindsight_instruction_returned = True
        self.hindsight_instruction = self.np_random.choice(hindsight_sentences, 1)[0]

    def reset_hi_and_ar(self):
        self.sim.bclient.removeUserDebugItem(self.action_repair_sim_id)

        self.ep_hindsight_instruction = False
        self.ep_hindsight_instruction_returned = False
        self.ep_action_repair = False
        self.ep_action_repair_returned = False
        self.ep_action_repair_success = False
        self.ep_delayed_ar_command = None

        # if both options are true, select one of them randomly
        if self.use_hindsight_instructions and self.use_action_repair:
            if self.np_random.random() < 0.5:
                self.ep_hindsight_instruction = self.np_random.random() < 0.5
            else:
                self.ep_action_repair = self.np_random.random() < 0.5
        elif self.use_hindsight_instructions:
            # generate hindsight instructions with a 25% chance
            self.ep_hindsight_instruction = self.np_random.random() < 0.25
        elif self.use_action_repair:
            # generate action repair commands with a 50% chance
            self.ep_action_repair = self.np_random.random() < 0.50

        if self.ep_hindsight_instruction:
            self.total_hindsight_instruction_episodes += 1
        if self.ep_action_repair:
            self.total_action_repair_episodes += 1

    def get_task_metrics(self) -> Dict:
        """ Returns a dict of task-specific metrics """
        metrics = {}
        if self.use_action_repair:
            metrics["AR_episodes"] = self.total_action_repair_episodes
            if self.total_action_repair_episodes:
                metrics["AR_success_rate"] = round(self.action_repair_success_ctr / self.total_action_repair_episodes,
                                                   2)
        if self.use_hindsight_instructions:
            metrics["HI_episodes"] = self.total_hindsight_instruction_episodes
            if self.total_hindsight_instruction_episodes:
                metrics["HI_discovery_rate"] = round(
                    self.discovered_hindsight_instruction_ctr / self.total_hindsight_instruction_episodes, 2)
        return metrics
