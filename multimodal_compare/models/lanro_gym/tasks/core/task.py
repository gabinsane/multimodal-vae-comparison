from typing import Dict
import numpy as np
from lanro_gym.utils import goal_distance


class Task:
    distance_threshold: float = 0.05
    reward_type: str = "sparse"
    last_distance: float = 0
    object_size: float = 0.04
    goal: np.ndarray
    np_random: np.random.Generator

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        return self.goal.copy()

    def get_obs(self):
        """Return the observation associated to the task."""
        raise NotImplementedError

    def get_achieved_goal(self):
        """Return the achieved goal."""
        raise NotImplementedError

    def reset(self):
        """Reset the task"""
        raise NotImplementedError

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        return False

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        return 0

    def get_task_metrics(self) -> Dict:
        return {}
