from .object_properties import RGBCOLORS, SHAPES, SIZES, DUMMY, WEIGHTS
from .task_object import TaskObject
from .task_object_list import TaskObjectList
import numpy as np


def distinguishable_by_primary(goal_obj: TaskObject, non_goal_obj: TaskObject):
    if isinstance(goal_obj.primary, RGBCOLORS):
        return goal_obj.color != non_goal_obj.color
    elif isinstance(goal_obj.primary, SHAPES):
        return goal_obj.shape != non_goal_obj.shape
    elif isinstance(goal_obj.primary, SIZES):
        return goal_obj._size != non_goal_obj._size
    elif isinstance(goal_obj.primary, WEIGHTS):
        return goal_obj.weight != non_goal_obj.weight
    else:
        return False


def distinguishable_by_primary_or_secondary(goal_obj: TaskObject, non_goal_obj: TaskObject):
    primary_diff = distinguishable_by_primary(goal_obj, non_goal_obj)
    secondary_diff = False
    if isinstance(goal_obj.secondary, RGBCOLORS):
        secondary_diff = goal_obj.color != non_goal_obj.color
    elif isinstance(goal_obj.secondary, SHAPES):
        secondary_diff = goal_obj.shape != non_goal_obj.shape
    elif isinstance(goal_obj.secondary, SIZES):
        secondary_diff = goal_obj._size != non_goal_obj._size
    elif isinstance(goal_obj.secondary, WEIGHTS):
        secondary_diff = goal_obj.weight != non_goal_obj.weight
    return np.sum([primary_diff, secondary_diff]) > 0


def dummys_not_goal_props(goal_obj: TaskObject, non_goal_obj: TaskObject):
    dummy_props = []
    if non_goal_obj.has_dummy_color:
        dummy_props.append(non_goal_obj.get_color())
    if non_goal_obj.has_dummy_shape:
        dummy_props.append(non_goal_obj.get_shape())
    if non_goal_obj.has_dummy_size:
        dummy_props.append(non_goal_obj.get_size())
    if non_goal_obj.has_dummy_weight:
        dummy_props.append(non_goal_obj.get_weight())

    primary_dummy_same = goal_obj.primary in dummy_props
    secondary_dummy_same = goal_obj.secondary in dummy_props
    one_overlap = np.sum([primary_dummy_same, secondary_dummy_same]) < 2
    return one_overlap and distinguishable_by_primary_or_secondary(goal_obj, non_goal_obj)


def valid_task_object_combination(goal_obj: TaskObject, non_goal_obj: TaskObject):
    goal_primary = goal_obj.primary
    goal_secondary = goal_obj.secondary

    non_goal_primary = non_goal_obj.primary
    non_goal_secondary = non_goal_obj.secondary
    different_primary = (goal_primary != non_goal_primary)
    different_primary_secondary = (goal_primary != non_goal_secondary)

    if isinstance(goal_secondary, DUMMY):
        if isinstance(non_goal_secondary, DUMMY):
            primary_dummy_different = distinguishable_by_primary(goal_obj, non_goal_obj)
            return different_primary and primary_dummy_different
        else:
            return different_primary_secondary and different_primary
    elif isinstance(non_goal_secondary, DUMMY):
        return dummys_not_goal_props(goal_obj, non_goal_obj)
    else:
        different_secondary = (goal_secondary != non_goal_secondary)
        return distinguishable_by_primary_or_secondary(goal_obj, non_goal_obj) and (different_secondary
                                                                                    or different_primary_secondary)
