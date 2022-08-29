from enum import Enum
from collections import namedtuple


class NetworkTypes(Enum):
    UNSPECIFIED = -1
    CNN = 1
    TXTTRANSFORMER = 2
    FNN = 3

VaeOutput = namedtuple("vae_output", ["encoder_dists", "decoder_dists","latent_samples"])


class NetworkRoles(Enum):
    UNSPECIFIED = -1
    ENCODER = 1
    DECODER = 2
    MIXER = 3
