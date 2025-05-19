import os
from typing import Dict, List, Tuple, Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.core import Task
from lanro_gym.env import BaseEnv

gym.logger.set_level(40)

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])


class GoalEnv(BaseEnv):
    ep_end_goal_distance: List = []

    def __init__(self, sim: PyBulletSimulation, robot: PyBulletRobot, task: Task, obs_type: str = "state"):
        BaseEnv.__init__(self, sim, robot, task, obs_type)

        obs, _ = self.reset()
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=self.obs_low,
                                       high=self.obs_high,
                                       shape=obs["observation"].shape,
                                       dtype=np.float32),
                desired_goal=spaces.Box(low=self.obs_low,
                                        high=self.obs_high,
                                        shape=(3,1),
                                        dtype=np.float32),
                achieved_goal=spaces.Box(low=self.obs_low,
                                         high=self.obs_high,
                                         shape=(3,1),
                                         dtype=np.float32),
            ))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()
        task_obs = self.task.get_obs()
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = None
        desired_goal = None #self.task.get_goal()
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        truncated = False
        info = {
            "is_success": False,
        }
        terminated = bool(info["is_success"])
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        return obs, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.ep_end_goal_distance.append(self.task.last_distance)
        obs, info = super().reset(seed=seed, options=options)
        info["is_success"] = self.task.is_success(obs["achieved_goal"], obs["desired_goal"])
        return obs, info

    def get_metrics(self) -> Dict:
        return {"avg_terminal_goal_distance": round(np.mean(self.ep_end_goal_distance), 3)}
