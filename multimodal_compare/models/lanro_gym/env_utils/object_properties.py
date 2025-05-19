from enum import Enum
from lanro_gym.utils import scale_rgb


class DUMMY(Enum):
    OBJECT = 0, ['object']


class SHAPES(Enum):
    """ SHAPES enum class with all shapes with the corresponding object file id and words"""
    CUBE = 0, ["box", "block"],
    CUBOID = 1, ["brick", "oblong"],
    CYLINDER = 2, ["barrel", "tophat"],
    MUG = 3, ["mug"],
    SOAP = 4, ["soap"],
    LEMON = 5, ["lemon"],
    TOOTHPASTE = 6, ["toothpaste"],
    STAPLER = 7, ["stapler"],
    TEABOX = 8, ["tea box"],
    DRAWER = 9, ["drawer"],
    NOTHING = 10, ["nothing"],

class RGBCOLORS(Enum):
    """ RGBColors enum class with all colors defined as array of floats [0, 1]"""
    BLACK = scale_rgb([0, 0, 0]), ["ebony"]
    BLUE = scale_rgb([78.0, 121.0, 167.0]), ["azure"], [SHAPES.SOAP]
    BROWN = scale_rgb([156.0, 117.0, 95.0]), ["chocolate"]
    CYAN = scale_rgb([118.0, 183.0, 178.0]), ["teal"]
    GRAY = scale_rgb([186.0, 176.0, 172.0]), ["ashen"]
    GREEN = scale_rgb([89.0, 169.0, 79.0]), ["lime"], [SHAPES.TOOTHPASTE]
    PINK = scale_rgb([255.0, 157.0, 167.0]), ["coral"]
    ORANGE = scale_rgb([242.0, 142.0, 43.0]), ["apricot"]
    PURPLE = scale_rgb([176.0, 122.0, 161.0]), ["lilac"]
    RED = scale_rgb([255.0, 87.0, 89.0]), ["scarlet"], [SHAPES.MUG]
    WHITE = scale_rgb([255, 255, 255]), ["colorless"]
    YELLOW = scale_rgb([237.0, 201.0, 72.0]), ["amber"], [SHAPES.LEMON]


class WEIGHTS(Enum):
    LIGHT = 1, ['lightweight', 'lite']
    HEAVY = 4, ['heavyweight', 'massy']


class SIZES(Enum):
    SMALL = 0.03, ['little', 'tiny']
    MEDIUM = 0.04, ['midsize', 'moderate-size']
    BIG = 0.05, ['large', 'tall']
