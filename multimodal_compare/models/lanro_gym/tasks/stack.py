import numpy as np
from typing import Tuple
from lanro_gym.tasks.core import Task
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.scene import basic_scene
from lanro_gym.env_utils import RGBCOLORS


class Stack(Task):

    def __init__(
        self,
        sim: PyBulletSimulation,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
        goal_z_range: float = 0.0,
        num_obj: int = 2,
    ):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.goal_z_range = goal_z_range
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        assert num_obj <= 4, "Maximum number of objects is 4"
        self.obj_colors = [RGBCOLORS.RED, RGBCOLORS.GREEN, RGBCOLORS.BLUE, RGBCOLORS.YELLOW][:num_obj]
        self.num_obj = num_obj
        self.goal_offsets = [1, 3, 5, 7]
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def _create_scene(self):
        basic_scene(self.sim)

        for idx, obj_color in enumerate(self.obj_colors):
            self.sim.create_box(
                body_name=f"object{idx}",
                half_extents=[
                    self.object_size / 2,
                    self.object_size / 2,
                    self.object_size / 2,
                ],
                mass=2.0,
                position=[0.0, 0.0, self.object_size / 2],
                rgba_color=obj_color.value[0] + [1],
            )
            self.sim.create_sphere(
                body_name=f"target{idx}",
                radius=self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=[0.0, 0.0, self.object_size / 2],
                rgba_color=obj_color.value[0] + [0.3],
            )

    def get_obs(self) -> np.ndarray:
        observation = []
        for idx in range(self.num_obj):
            obj_key = f"object{idx}"
            # position, rotation of the object
            object_position = np.array(self.sim.get_base_position(obj_key))
            object_rotation = np.array(self.sim.get_base_rotation(obj_key))
            object_velocity = np.array(self.sim.get_base_velocity(obj_key))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity(obj_key))
            observation.extend([object_position, object_rotation, object_velocity, object_angular_velocity])
        return np.concatenate(observation)

    def get_achieved_goal(self) -> np.ndarray:
        achieved_goals = [self.sim.get_base_position(f"object{idx}") for idx in range(self.num_obj)]
        return np.concatenate(achieved_goals).copy()

    def _sample_objects(self) -> Tuple:
        obj_positions = [[0.0, 0.0, self.object_size / 2] +
                         self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                         for _ in range(len(self.obj_colors))]
        return tuple(obj_positions)

    def reset(self) -> None:
        self.goal = self._sample_goal()
        for idx, obj_pos in enumerate(self._sample_objects()):
            self.sim.set_base_pose(f"target{idx}", self.goal[3 * idx:3 * (idx + 1)].tolist(), [0, 0, 0, 1])
            self.sim.set_base_pose(f"object{idx}", obj_pos.tolist(), [0, 0, 0, 1])

    def _sample_goal(self) -> np.ndarray:
        goal = []
        goal_noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        for idx in range(self.num_obj):
            # if we have the first object and use goal_z_range, assign a goal
            # height of 0.0 at least 30% of the time
            if idx == 0 and self.goal_z_range > 0 and self.np_random.random() < 0.3:
                goal_noise[0] = 0.0

            goal.append([0.0, 0.0, self.goal_offsets[idx] * self.object_size / 2] +
                        goal_noise) # with z offset factor for stacking goals

        return np.concatenate((goal)).copy()
