import torch
from models.mmvae_base import TorchMMVAE
from models.vae import VAE

class ExampleModel(TorchMMVAE):
    def __init__(self):
        super().__init__()
        self.vaes["mod_1"] = VAE("CNN", "CNN", [3,64,64], 64)
        self.vaes["mod_2"] = VAE("CNN", "CNN", [3,64,64], 64)


def test_torch_mmvae_encode():
    print("Hello")
    mmvae = ExampleModel()
    inputs = {"mod_1": torch.rand((32, 3,64,64)), "mod_2":torch.rand((32, 3,64,64))}
    qz_xs = mmvae.encode(inputs, 1)
    assert isinstance(qz_xs, dict)
    assert len(qz_xs.keys()) == len(mmvae.vaes.keys())
    for dist in qz_xs.values():
        assert isinstance(dist, torch.dist)
        assert dist.loc.shape == 64


