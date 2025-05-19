from __future__ import annotations
from typing import Tuple, Union, Any
import numpy as np
from lanro_gym.env_utils.object_properties import WEIGHTS
from lanro_gym.utils import get_one_hot_list
from lanro_gym.env_utils import RGBCOLORS, SHAPES, SIZES, DUMMY
import random
import os

def get_default_enum_index(default_enum_cls, default_enum):
    enum_list = [e for e in default_enum_cls]
    enum_idx = enum_list.index(default_enum)
    return default_enum, enum_idx


class TaskObject:

    def __init__(self, sim, primary=None, secondary=None, onehot_idx=None, sec_onehot_idx=None, obj_mass: int = 2):
        self.sim = sim

        self.primary = primary
        self.secondary = DUMMY.OBJECT
        self.obj_mass = obj_mass

        self.onehot_colors = get_one_hot_list(len(RGBCOLORS))
        self.onehot_shapes = get_one_hot_list(len(SHAPES))
        self.onehot_sizes = get_one_hot_list(len(SIZES))
        self.onehot_weights = get_one_hot_list(len(WEIGHTS))

        self.onehot_idx_colors = None
        self.onehot_idx_shapes = None
        self.onehot_idx_sizes = None
        self.onehot_idx_weights = None

        self.has_dummy_size = False
        self.has_dummy_color = False
        self.has_dummy_shape = False
        self.has_dummy_weight = False

        if type(primary) == type(secondary):
            raise ValueError("primary and secondary must be different object types")

        if isinstance(primary, SIZES):
            self.onehot_idx_sizes = onehot_idx
            self._size = primary
            self.object_size = primary.value[0]
        elif isinstance(secondary, SIZES):
            self.secondary = secondary
            self._size = secondary
            self.object_size = secondary.value[0]
            self.onehot_idx_sizes = sec_onehot_idx
        else:
            self.has_dummy_size = True
            dummy_enum, self.onehot_idx_sizes = get_default_enum_index(SIZES, SIZES.MEDIUM)
            self._size = dummy_enum
            self.object_size = dummy_enum.value[0]

        if isinstance(primary, RGBCOLORS):
            self.onehot_idx_colors = onehot_idx
            self.color = primary
        elif isinstance(secondary, RGBCOLORS):
            self.secondary = secondary
            self.color = secondary
            self.onehot_idx_colors = sec_onehot_idx
        else:
            self.has_dummy_color = True
            self.color, self.onehot_idx_colors = get_default_enum_index(RGBCOLORS, RGBCOLORS.RED)

        if isinstance(primary, SHAPES):
            self.onehot_idx_shapes = onehot_idx
            self.shape = primary
        elif isinstance(secondary, SHAPES):
            self.secondary = secondary
            self.shape = secondary
            self.onehot_idx_shapes = sec_onehot_idx
        else:
            self.has_dummy_shape = True
            _dummy_enum, self.onehot_idx_shapes = get_default_enum_index(SHAPES, random.choice(self.color.value[-1]))
            self.shape = _dummy_enum

        if isinstance(primary, WEIGHTS):
            self.onehot_idx_weights = onehot_idx
            self.weight = primary
            self.obj_mass = self.obj_mass * primary.value[0]
        elif isinstance(secondary, WEIGHTS):
            self.secondary = secondary
            self.weight = secondary
            self.onehot_idx_weights = sec_onehot_idx
            self.obj_mass = self.obj_mass * secondary.value[0]
        else:
            self.has_dummy_weight = True
            _dummy_enum, self.onehot_idx_weights = get_default_enum_index(WEIGHTS, WEIGHTS.LIGHT)
            self.weight = _dummy_enum
            self.obj_mass = self.obj_mass * _dummy_enum.value[0]

    def __hash__(self):
        return hash(self.primary) ^ hash(self.secondary)

    def load(self, object_body_key):
        cw = "./models/" if not "models" in os.getcwd() else ""
        if self.shape == SHAPES.CUBE:
            self.sim.create_box(
                body_name=object_body_key,
                half_extents=[
                    self.object_size / 2,
                    self.object_size / 2,
                    self.object_size / 2,
                ],
                mass=self.obj_mass,
                position=[0.0, 0.0, self.object_size / 2],
                rgba_color=self.color.value[0] + [1],
            )
        elif self.shape == SHAPES.CUBOID:
            self.sim.create_box(
                body_name=object_body_key,
                half_extents=[
                    self.object_size / 2 * 2,
                    self.object_size / 2 * 0.75,
                    self.object_size / 2 * 0.75,
                ],
                mass=self.obj_mass,
                position=[0.0, 0.0, self.object_size / 2],
                rgba_color=self.color.value[0] + [1],
            )
        elif self.shape == SHAPES.CYLINDER:
            self.sim.create_cylinder(
                body_name=object_body_key,
                radius=self.object_size * 0.5,
                height=self.object_size * 0.75,
                mass=self.obj_mass * 3,
                position=[0.0, 0.0, self.object_size / 2],
                rgba_color=self.color.value[0] + [1],
                lateral_friction=1.0,
                spinning_friction=0.005,
            )

        elif self.shape == SHAPES.SOAP:
            id = self.sim.loadURDF(body_name=object_body_key,
                                   fileName=".{}/lanro_gym/objects_urdfs/soap.urdf".format(cw),
                                   basePosition=[0,0,0.1])
        elif self.shape == SHAPES.MUG:
            id = self.sim.loadURDF(body_name=object_body_key,
                                   fileName=".{}/lanro_gym/objects_urdfs/mug.urdf".format(cw)
                                   )
        elif self.shape == SHAPES.LEMON:
            id = self.sim.loadURDF(body_name=object_body_key,
                                   fileName=".{}/lanro_gym/objects_urdfs/lemon.urdf".format(cw))
        elif self.shape == SHAPES.NOTHING:
            self.sim.create_cylinder(
                body_name=object_body_key,
                radius=0.0000001,
                height=0.0000001,
                mass=0.0000001,
                position=[0.0, 0.0, 0],
                rgba_color=self.color.value[0] + [0],
                lateral_friction=1.0,
                spinning_friction=0.005,
            )
        elif self.shape == SHAPES.TOOTHPASTE:
            id = self.sim.loadURDF(body_name=object_body_key,
                                   fileName=".{}/lanro_gym/objects_urdfs/toothpaste.urdf".format(cw))
        elif self.shape == SHAPES.STAPLER:
            id = self.sim.loadURDF(body_name=object_body_key,
                                   fileName=".{}/lanro_gym/objects_urdfs/stapler.urdf".format(cw))
        elif self.shape == SHAPES.TEABOX:
            id = self.sim.loadURDF(body_name=object_body_key,
                                   fileName=".{}/lanro_gym/objects_urdfs/teabox.urdf".format(cw))


    def load_test(self, object_body_key):
        self.object_size /= 10000
        self.sim.create_cylinder(
            body_name=object_body_key,
            radius=self.object_size * 0.5,
            height=self.object_size * 0.75,
            mass=self.obj_mass * 3,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=self.color.value[0] + [1],
            lateral_friction=1.0,
            spinning_friction=0.005,
        )

    def get_properties(self) -> Tuple:
        return self.primary, self.secondary

    def get_color(self) -> RGBCOLORS:
        return self.color

    def get_shape(self) -> Union[SHAPES, Any]:
        return self.shape

    def get_size(self) -> Union[SIZES, Any]:
        return self._size

    def get_weight(self) -> Union[WEIGHTS, Any]:
        return self.weight

    def __eq__(self, __o: TaskObject) -> bool:
        """Two task objects are equal if they have at least three properties in common."""
        same_color = self.color == __o.color
        same_shape = self.shape == __o.shape
        same_size = self._size == __o._size
        same_weight = self.weight == __o.weight
        return np.sum([same_color, same_shape, same_size, same_weight]) > 2

    def get_onehot(self):
        size_onehot = self.onehot_sizes[self.onehot_idx_sizes]
        color_onehot = self.onehot_colors[self.onehot_idx_colors]
        shape_onehot = self.onehot_shapes[self.onehot_idx_shapes]
        weight_onehot = self.onehot_weights[self.onehot_idx_weights]
        return np.concatenate([size_onehot, color_onehot, shape_onehot, weight_onehot])

    def get_color_strs(self):
        return [list(RGBCOLORS)[self.onehot_idx_colors].name.lower()] + list(RGBCOLORS)[self.onehot_idx_colors].value[1]

    def get_shape_strs(self):
        return [list(SHAPES)[self.onehot_idx_shapes].name.lower()] + list(SHAPES)[self.onehot_idx_shapes].value[1]

    def get_size_strs(self):
        return [list(SIZES)[self.onehot_idx_sizes].name.lower()] + list(SIZES)[self.onehot_idx_sizes].value[1]

    def get_weight_strs(self):
        return [list(WEIGHTS)[self.onehot_idx_weights].name.lower()] + list(WEIGHTS)[self.onehot_idx_weights].value[1]
