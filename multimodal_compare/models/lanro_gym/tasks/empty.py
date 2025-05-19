import numpy as np
from typing import Callable
from lanro_gym.tasks.core import Task
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.scene import basic_scene
from lanro_gym.env_utils import RGBCOLORS
import random

class Empty(Task):

    def __init__(self,
                 sim: PyBulletSimulation,
                 get_ee_position: Callable[[], np.ndarray],
                 reward_type: str = "sparse",
                 distance_threshold: float = 0.025,
                 goal_range: float = 0.1):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-0.2, -0.2, 0.03])
        self.goal_range_high = np.array([0, 0, 0.03])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        basic_scene(self.sim)

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        return self.get_ee_position()

    def reset(self) -> None:
        self._create_scene()


    def _sample_goal(self, v) -> np.ndarray:
        if v == 0:
            return self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        else:
            return self.np_random.uniform([0, 0, 0.03], [0.2, 0.2, 0.03])