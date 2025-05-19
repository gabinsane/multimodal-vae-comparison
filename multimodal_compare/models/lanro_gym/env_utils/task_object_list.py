from enum import Enum
from typing import Dict, List
from lanro_gym.utils import get_prop_combinations
from lanro_gym.env_utils import RGBCOLORS, SHAPES, WEIGHTS, SIZES, TaskObject


class TaskObjectList:

    def __init__(self,
                 sim,
                 color_mode: bool = False,
                 shape_mode: bool = False,
                 weight_mode: bool = False,
                 size_mode: bool = False):
        self.sim = sim
        # default colors
        concept_list: List[Enum] = []#[RGBCOLORS.RED, RGBCOLORS.GREEN, RGBCOLORS.BLUE, RGBCOLORS.YELLOW]
        if color_mode:
            # extend range of colors
            concept_list.extend([
                RGBCOLORS.YELLOW,
                RGBCOLORS.PURPLE,
                RGBCOLORS.ORANGE,
                RGBCOLORS.PINK,
                RGBCOLORS.CYAN,
                RGBCOLORS.BROWN,
            ])
        # shape mode combinations
        if shape_mode:
            concept_list.extend([SHAPES.DRAWER]) # SHAPES.TOOTHPASTE, SHAPES.STAPLER, SHAPES.TEABOX, SHAPES.SOAP,
            #concept_list.extend([SHAPES.CUBE, SHAPES.CUBOID, SHAPES.CYLINDER])
        if weight_mode:
            concept_list.extend([WEIGHTS.HEAVY, WEIGHTS.LIGHT])
        if size_mode:
            concept_list.extend([SIZES.SMALL, SIZES.MEDIUM, SIZES.BIG])

        self.objects = self.setup(concept_list)

    def setup(self, concept_list) -> List[TaskObject]:
        objects = []

        # add single property to task
        for concept in concept_list:
            _args = self.get_task_obj_args({}, concept)
            #print(_args)
            objects.append(TaskObject(self.sim, **_args))

        concept_tuple_list = get_prop_combinations(concept_list)
        if len(concept_tuple_list):
            for concept_tuple in concept_tuple_list:
                prop1 = concept_tuple[0]
                prop2 = concept_tuple[1]
                _args = self.get_task_obj_args({}, prop1)
                _args = self.get_task_obj_args(_args, prop2, primary=False)
                objects.append(TaskObject(self.sim, **_args))
        return objects

    @staticmethod
    def get_task_obj_args(_args, prop, primary=True) -> Dict:
        if isinstance(prop, SIZES):
            enum = SIZES
        elif isinstance(prop, SHAPES):
            enum = SHAPES
        elif isinstance(prop, WEIGHTS):
            enum = WEIGHTS
        elif isinstance(prop, RGBCOLORS):
            enum = RGBCOLORS
        prop_list = [sz for sz in enum]
        prop_idx = prop_list.index(prop)
        if primary:
            _args['primary'] = prop
            _args['onehot_idx'] = prop_idx
        else:
            _args['secondary'] = prop
            _args['sec_onehot_idx'] = prop_idx
        return _args

    def get_obj_properties(self, objects=None) -> List:
        if objects is None:
            objects = self.objects
        return [obj.get_properties() for obj in objects]

    def __getitem__(self, index) -> TaskObject:
        return self.objects[index]

    def __len__(self) -> int:
        return len(self.objects)
