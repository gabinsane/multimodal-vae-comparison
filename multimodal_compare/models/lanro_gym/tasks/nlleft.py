import numpy as np
from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.core import LanguageTask
from lanro_gym.language_utils import create_commands


class NLLeft(LanguageTask):

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 obj_xy_range: float = 0.2,
                 num_obj: int = 2,
                 min_goal_height: float = 0.0,
                 max_goal_height: float = 0.1,
                 use_hindsight_instructions: bool = False,
                 use_action_repair: bool = False,
                 delay_action_repair: bool = False,
                 use_negations_action_repair: bool = False,
                 use_synonyms: bool = False,
                 mode: str = 'Color'):
        super().__init__(sim, robot, mode, use_hindsight_instructions, use_action_repair, delay_action_repair,
                         use_negations_action_repair, num_obj, use_synonyms)
        self.max_goal_height = max_goal_height
        self.test_only = False
        self.num_obj = num_obj
        self.min_goal_height = min_goal_height
        self.obj_range_low = np.array([-0.05, -0.2, 0])
        self.obj_range_high = np.array([obj_xy_range, obj_xy_range, 0])
        self.action_verbs = ["move left", "shift left", "drag left"]
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def reset(self) -> None:
        self.sample_task_objects(testing=self.test_only)
        if self.num_obj > 1:
            for idx, obj_pos in zip(self.obj_indices_selection, self._sample_objects()):
                self.sim.set_base_pose(f"object{idx}", obj_pos, [0, 0, 0, 1])
        if self.num_obj == 1:
            randpose = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            self.sim.set_base_pose(f"object{self.obj_indices_selection[0]}", randpose, [0, 0, 0, 1])
        self._sample_goal()
        self.ep_height_threshold = self.np_random.uniform(low=self.min_goal_height, high=self.max_goal_height)
        # similar to the pick and place task, the goal height is 0 at least 30% of the time
        if self.np_random.random() < 0.3:
            self.ep_height_threshold = 0
        self.reset_hi_and_ar()

    def grasped_and_lifted(self, obj_body_key):
        obj_pos = np.array(self.sim.get_base_position(obj_body_key))
        hit_obj_id = self.robot.gripper_ray_obs()[0]
        obj_id = self.sim.get_object_id(obj_body_key)
        all_fingers_have_contact = np.all(self.get_contact_with_fingers(obj_body_key))
        achieved_min_height = obj_pos[-1] > self.ep_height_threshold
        inside_gripper = hit_obj_id == obj_id
        return all_fingers_have_contact and achieved_min_height and inside_gripper

    def is_success(self):
        return self.grasped_and_lifted(self.goal_object_body_key)

    def compute_reward(self) -> float:
        if self.is_success():
            return self.generate_action_repair_or_success()
        elif self.ep_hindsight_instruction and not self.ep_hindsight_instruction_returned:
            for other_object_idx in self.non_goal_body_indices:
                # if grasped with both fingers and being at a certain height
                if self.grasped_and_lifted(f"object{other_object_idx}"):
                    self.generate_hindsight_instruction(other_object_idx)
                    return -10.
        elif self.ep_action_repair and not self.ep_action_repair_returned:
            for other_object_idx in self.non_goal_body_indices:
                # if grasped with both fingers and being at a certain height
                if self.grasped_and_lifted(f"object{other_object_idx}"):
                    if self.use_negations_action_repair and self.np_random.random() < 0.5:
                        # action correction with negation
                        # "lift the red object" -> lifts green object -> correction "no not the green object"
                        target_property_tuple = self.task_object_list.objects[other_object_idx].get_properties()
                        repair_commands = create_commands("negation",
                                                          target_property_tuple,
                                                          use_synonyms=self.use_synonyms)
                    else:
                        # additional feedback for the goal object, lifting a wrong object
                        # "lift the red block" -> *lifts green block* -> "the red block!"
                        target_property_tuple = self.task_object_list.objects[self.goal_obj_idx].get_properties()
                        repair_commands = create_commands("repair",
                                                          target_property_tuple,
                                                          use_synonyms=self.use_synonyms)
                    self.merge_instruction_action_repair(repair_commands)
                    return -1.0
        return -1.0
