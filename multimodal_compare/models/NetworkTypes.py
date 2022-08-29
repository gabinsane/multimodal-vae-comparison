from enum import Enum
from collections import namedtuple

class NetworkTypes(Enum):
    CNN = 1
    TXTTRANSFORMER = 2
    FNN = 3

VaeOutput = namedtuple("vae_output", ["encoder_dists", "decoder_dists","latent_samples"])
