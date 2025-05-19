from lanro_gym.robots import PyBulletRobot
import numpy as np
from lanro_gym.tasks.core import LanguageTask
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.utils import goal_distance
from lanro_gym.language_utils import create_commands


class NLReach(LanguageTask):

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 obj_xy_range: float = 0.3,
                 num_obj: int = 2,
                 use_hindsight_instructions: bool = False,
                 use_action_repair: bool = False,
                 delay_action_repair: bool = False,
                 use_negations_action_repair: bool = False,
                 use_synonyms: bool = False,
                 mode: str = 'Color'):
        super().__init__(sim, robot, mode, use_hindsight_instructions, use_action_repair, delay_action_repair,
                         use_negations_action_repair, num_obj, use_synonyms)
        self.obj_range_low = np.array([-0.05, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.test_only = True
        self.action_verbs = ["touch", "reach", "contact"]
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def reset(self) -> None:
        self.sample_task_objects(testing=self.test_only)
        self.obj_init_pos = self._sample_objects()
        for idx, obj_pos in zip(self.obj_indices_selection, self.obj_init_pos):
             self.sim.set_base_pose(f"object{idx}", obj_pos.tolist(), [0, 0, 0, 1])
        self._sample_goal()
        self.reset_hi_and_ar()

    def is_success(self):
        # NOTE: objects should stay in place with maximum positional change \eta to initial position
        current_obj_pos = np.concatenate(
            [np.array(self.sim.get_base_position(f"object{idx}")) for idx in self.obj_indices_selection])
        close_to_init_pos = goal_distance(np.concatenate(self.obj_init_pos), current_obj_pos) < 0.025
        # check if ticked correct object
        return np.any(self.get_contact_with_fingers(self.goal_object_body_key)) and close_to_init_pos

    def compute_reward(self) -> float:
        if self.is_success():
            return self.generate_action_repair_or_success()
        elif self.ep_hindsight_instruction and not self.ep_hindsight_instruction_returned:
            for other_object_idx in self.non_goal_body_indices:
                _non_goal_body = f"object{other_object_idx}"
                # if touched with at least one finger
                if np.any(self.get_contact_with_fingers(_non_goal_body)):
                    self.generate_hindsight_instruction(other_object_idx)
                    return -10.
        elif self.ep_action_repair and not self.ep_action_repair_returned:
            for other_object_idx in self.non_goal_body_indices:
                _non_goal_body = f"object{other_object_idx}"
                # if touched with at least one finger
                if np.any(self.get_contact_with_fingers(_non_goal_body)):
                    if self.use_negations_action_repair and self.np_random.random() < 0.5:
                        # action correction with negation
                        # "reach the red object" -> reaches green object -> correction "no not the green object"
                        target_property_tuple = self.task_object_list.objects[other_object_idx].get_properties()
                        repair_commands = create_commands("negation",
                                                          target_property_tuple,
                                                          use_synonyms=self.use_synonyms)
                    else:
                        # additional feedback for the goal object, touching a wrong object
                        # "touch the red block" -> *touches green block* -> "the red block!"
                        target_property_tuple = self.task_object_list.objects[self.goal_obj_idx].get_properties()
                        repair_commands = create_commands("repair",
                                                          target_property_tuple,
                                                          use_synonyms=self.use_synonyms)
                    self.merge_instruction_action_repair(repair_commands)
                    return -1.0
        return -1.0
