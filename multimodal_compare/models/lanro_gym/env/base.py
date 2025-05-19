import os
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.core import LanguageTask, Task

gym.logger.set_level(40)

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])


class BaseEnv(gym.Env):
    """
    BaseEnv is a goal-conditoned Gym environment that inherits from `gym.Env`.
    """
    obs_low: float = -240.0
    obs_high: float = 240.0

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 task: Union[Task, LanguageTask],
                 obs_type: str = "state"):
        self.sim = sim
        self.metadata = {"render_modes": ["human", "rgb_array"], 'video.frames_per_second': int(np.round(1 / sim.dt))}
        self.reward_range = (-1.0, 0.0)
        self.robot = robot
        self.action_space = self.robot.action_space
        self.task = task
        self.obs_type = obs_type

    def close(self) -> None:
        self.sim.close()

    def _get_obs(self):
        raise NotImplementedError

    def getKeyboardEvents(self) -> Dict[int, int]:
        return self.sim.bclient.getKeyboardEvents()

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        info = {"is_success": False}
        return self._get_obs(), info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.task.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode="human"):
        return self.sim.render(mode)
