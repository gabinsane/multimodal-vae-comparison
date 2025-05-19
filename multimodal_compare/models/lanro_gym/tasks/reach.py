import numpy as np
from typing import Callable
from lanro_gym.tasks.core import Task
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.scene import basic_scene
from lanro_gym.env_utils import RGBCOLORS
import random

class Reach(Task):

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
            self.sim.place_visualizer()

    def _create_scene(self) -> None:
        basic_scene(self.sim)
        colors = [RGBCOLORS.RED, RGBCOLORS.BLUE, RGBCOLORS.GREEN, RGBCOLORS.YELLOW]
        c, num = None, None
        c_b, num_b = None, None
        for o in ["target", "distractor"]:
            if num is not None:
                num_b = num
                c_b = c
            num = random.randint(0, 2)
            c = random.choice(colors)
            if c == c_b and num == num_b:
                try:
                    colors.remove(c_b)
                    c = random.choice(colors)
                except:
                    pass
            if num == 0:
                print("adding sphere")
                self.sim.create_sphere(
                    body_name=o,
                    radius=self.distance_threshold,
                    mass=0.0,
                    ghost=False,
                    position=[0.0, 0.0, 0.0],
                    rgba_color=c.value[0] + [1],
                )
            elif num == 1:
                print("adding box")
                self.sim.create_box(
                    body_name=o,
                    half_extents= [0.05 / 2, 0.05 / 2, 0.05 / 2],
                    mass=0.0,
                    ghost=True,
                    position=[0.0, 0.0, 0.0],
                    rgba_color=c.value[0] + [1],
                )
            else:
                print("adding cylinder")
                self.sim.create_cylinder(
                    body_name=o,
                    radius=self.distance_threshold,
                    height=0.05,
                    mass=0.0,
                    ghost=True,
                    position=[0.0, 0.0, 0.0],
                    rgba_color=c.value[0] + [1],
                )

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        return self.get_ee_position()

    def reset(self) -> None:
        self.sim.remove_body("target")
        self.sim.remove_body("distractor")
        self._create_scene()
        order = random.choice([[0,1], [1,0]])
        self.goal = self._sample_goal(order[0])
        self.sim.set_base_pose("target", self.goal.tolist(), [0, 0, 0, 1])
        self.sim.set_base_pose("distractor", self._sample_goal(order[1]).tolist(), [0, 0, 0, 1])

    def _sample_goal(self, v) -> np.ndarray:
        if v == 0:
            return self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        else:
            return self.np_random.uniform([0, 0, 0.03], [0.2, 0.2, 0.03])