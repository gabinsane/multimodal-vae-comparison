from enum import Enum
import itertools
from typing import List, Set, Tuple
import numpy as np
import random


def get_prop_combinations(stream) -> Set[Tuple]:
    combinations = itertools.permutations(stream, 2)
    comblist = [c for c in combinations]

    def filter_same_prop_type(x) -> bool:
        return type(x[0]) != type(x[1])

    filtered_combinations = [e for e in filter(filter_same_prop_type, comblist)]
    assert len(comblist) >= len(filtered_combinations)
    return set(filtered_combinations)


def expand_enums(concept_stream: List) -> List:
    objects = []
    for concept in concept_stream:
        if isinstance(concept, tuple):
            for concept_prod in itertools.product(*concept):
                objects.append(concept_prod)
        else:
            for prop in concept:
                objects.append(prop)
    return objects


def get_random_enum_with_exceptions(enum_cls, exclude) -> Tuple[Enum, int]:
    en_list = [e for e in enum_cls]
    en = random.choice([_e for _e in en_list if _e not in exclude])
    en_idx = en_list.index(en)
    return en, en_idx


def scale_rgb(rgb_lst: List[float]) -> List[float]:
    return [_color / 255.0 for _color in rgb_lst]


def get_one_hot_list(total_items: int) -> np.ndarray:
    """Create an array with `total_items` one-hot vectors."""
    return np.eye(total_items)


def goal_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    assert vec1.shape == vec2.shape, "mismatch of vector shapes"
    return np.linalg.norm(vec1 - vec2, axis=-1)


def post_process_camera_pixel(px, _height: int, _width: int) -> np.ndarray:
    rgb_array = np.array(px, dtype=np.uint8).reshape(_height, _width, 4)
    return rgb_array[:, :, :3]


def gripper_camera(bullet_client, projectionMatrix, pos, orn, imgsize: int = 84, mode: str = 'ego') -> np.ndarray:
    mode = "static"
    if mode == 'static':
        pos = [0.1, 0, 0]
        distance = 0.5
        yaw = 90
        pitch = -60
    elif mode == 'ego':
        distance = 0.5
        yaw = -45
        pitch = -30
    viewMatrix = bullet_client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=pos,
                                                                 distance=distance,
                                                                 yaw=yaw,
                                                                 pitch=pitch,
                                                                 roll=0,
                                                                 upAxisIndex=2)
    (_, _, px, _, _) = bullet_client.getCameraImage(imgsize,
                                                    imgsize,
                                                    viewMatrix,
                                                    projectionMatrix,
                                                    flags=bullet_client.ER_NO_SEGMENTATION_MASK,
                                                    shadow=0,
                                                    renderer=bullet_client.ER_BULLET_HARDWARE_OPENGL)
    return post_process_camera_pixel(px, imgsize, imgsize)


def environment_camera(bullet_client, projectionMatrix, viewMatrix, width: int = 500, height: int = 500) -> np.ndarray:
    (_, _, px, _, _) = bullet_client.getCameraImage(width,
                                                    height,
                                                    viewMatrix,
                                                    projectionMatrix,
                                                    flags=bullet_client.ER_NO_SEGMENTATION_MASK,
                                                    renderer=bullet_client.ER_BULLET_HARDWARE_OPENGL)
    return post_process_camera_pixel(px, width, height)
