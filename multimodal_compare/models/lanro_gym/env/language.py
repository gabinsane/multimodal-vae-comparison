import os
from typing import Dict, Set, Tuple, Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from lanro_gym.language_utils import Vocabulary, parse_instructions
from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.core import LanguageTask
from lanro_gym.env_utils import SHAPES, RGBCOLORS, SIZES, WEIGHTS
from lanro_gym.env import BaseEnv

gym.logger.set_level(40)

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])


class LanguageEnv(BaseEnv):
    """
    RobotLanguageEnv is a language-conditioned implementation of `BaseEnv` with a language-specific Gym API.
    """
    discovered_word_idxs: Set = set()

    def __init__(self, sim: PyBulletSimulation, robot: PyBulletRobot, task: LanguageTask, obs_type: str = "state"):
        BaseEnv.__init__(self, sim, robot, task, obs_type)

        instruction_list = self.task.get_all_instructions()
        if DEBUG:
            print("AMOUNT OF INSTRUCTIONS", len(instruction_list))
        self.word_list, self.max_instruction_len = parse_instructions(instruction_list)
        self.vocab = Vocabulary(self.word_list)
        obs, _ = self.reset()
        self.compute_reward = self.task.compute_reward

        instruction_index_space = spaces.Box(0, len(self.vocab), shape=(self.max_instruction_len, ), dtype=np.uint16)
        if self.obs_type == "state":
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(low=self.obs_low,
                                           high=self.obs_high,
                                           shape=obs["observation"].shape,
                                           dtype=np.float32),
                    instruction=instruction_index_space,
                ))
        elif self.obs_type == "pixel":
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(low=0, high=255, shape=obs['observation'].shape, dtype=np.uint8),
                    instruction=instruction_index_space,
                ))

    def get_vocab_by_properties(self):
        vocab_properties = {}
        for word in self.vocab.word2idx.keys():
            if word in np.concatenate([shape.value[1] for shape in SHAPES]):
                vocab_properties[word] = 'shape'
            elif word in [shape.name.lower() for shape in RGBCOLORS]:
                vocab_properties[word] = 'color'
            elif word in np.concatenate([_size.value[1] for _size in SIZES]):
                vocab_properties[word] = 'size'
            elif word in np.concatenate([_weight.value[1] for _weight in WEIGHTS]):
                vocab_properties[word] = 'size'
            elif word in self.task.action_verbs:
                vocab_properties[word] = 'action'
            else:
                vocab_properties[word] = 'none'
        return vocab_properties

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()
        task_obs = self.task.get_obs()

        if self.obs_type == "pixel":
            observation = self.robot.get_camera_img().copy()
        else:
            observation = np.concatenate([robot_obs, task_obs])
            if self.sim.render_on:
                _ = self.robot.get_camera_img()

        current_goal_string = self.task.get_goal()
        word_indices = self.encode_instruction(self.pad_instruction(current_goal_string))

        return {"observation": observation.copy(), "instruction": word_indices}

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info["is_success"] = self.task.is_success()
        return obs, info

    def pad_instruction(self, goal_string) -> str:
        _pad_diff = self.max_instruction_len - len(goal_string.split(' '))
        if _pad_diff:
            goal_string += ' ' + ' '.join(['<pad>'] * _pad_diff)
        return goal_string

    def get_vocab(self) -> Vocabulary:
        return self.vocab

    def get_max_instruction_len(self) -> int:
        return self.max_instruction_len

    def encode_instruction(self, instruction: str) -> np.ndarray:
        word_indices = [self.vocab.word_to_idx(word) for word in instruction.split(' ')]
        return np.array(word_indices)

    def decode_instruction(self, instruction_embedding) -> str:
        words = [self.vocab.idx_to_word(idx) for idx in instruction_embedding]
        return ' '.join(words)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.set_action(action)
        self.sim.step()
        self.task.return_delayed_action_repair()
        obs = self._get_obs()
        self.discovered_word_idxs.update(obs['instruction'])
        info = {
            "is_success": self.task.is_success(),
        }
        reward = self.compute_reward()
        terminated = bool(info["is_success"])
        truncated = False
        # HI created, add to info dict and terminate episode
        if reward == -10.0:
            terminated = True
            h_instr = self.pad_instruction(self.task.hindsight_instruction)
            info['hindsight_instruction_language'] = h_instr
            info['hindsight_instruction'] = self.encode_instruction(h_instr)
            self.discovered_word_idxs.update(info['hindsight_instruction'])
            # NOTE: set reward to normal punishment, as we like, e.g., NLReach
            # and NLReachHI to behave the same way
            reward = -1.0

        return obs, reward, terminated, truncated, info

    def get_metrics(self) -> Dict[str, Any]:
        """ Returns a dict of environment metrics"""
        return {
            "vocab_discovery_rate": round(len(self.discovered_word_idxs) / (len(self.vocab) - 1), 2),
            **self.task.get_task_metrics(),
        }
