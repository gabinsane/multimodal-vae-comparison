from enum import Enum


class NetworkTypes(Enum):
    UNSPECIFIED = -1
    CNN = 1
    TXTTRANSFORMER = 2
    FNN = 3
    TRANSFORMER = 4
    DCNN = 5


class NetworkRoles(Enum):
    UNSPECIFIED = -1
    ENCODER = 1
    DECODER = 2
    MIXER = 3
