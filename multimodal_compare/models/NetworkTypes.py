from enum import Enum
from collections import namedtuple


class NetworkTypes(Enum):
    UNSPECIFIED = -1
    CNN = 1
    TXTTRANSFORMER = 2
    FNN = 3
    TRANSFORMER = 4
    DCNN = 5

fields = ["encoder_dists", "decoder_dists","latent_samples", "single_latents"]
VaeOutput = namedtuple("vae_output", fields, defaults=(None,) * len(fields))


class NetworkRoles(Enum):
    UNSPECIFIED = -1
    ENCODER = 1
    DECODER = 2
    MIXER = 3
