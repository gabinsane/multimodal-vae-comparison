import numpy as np
from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.core import LanguageTask
from lanro_gym.language_utils import create_commands


class NLEmpty(LanguageTask):

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
        self.min_goal_height = min_goal_height
        self.test_only = False
        self.num_obj = num_obj
        self.obj_range_low = np.array([-0.05, -0.2, 0])
        self.obj_range_high = np.array([obj_xy_range, obj_xy_range, 0])
        self.action_verbs = ["lift", "raise", "hoist"]
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def reset(self) -> None:
        self.reset_hi_and_ar()

    def is_success(self):
        return self.grasped_and_lifted(self.goal_object_body_key)

    def compute_reward(self) -> float:
        return -1.0
