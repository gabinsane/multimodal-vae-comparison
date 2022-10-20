import torch
from models.mmvae_base import TorchMMVAE
from models.vae import VAE
import inspect
from models.output_storage import VAEOutput

class ExampleModel(TorchMMVAE):
    def __init__(self, n_latents, obj):
        super().__init__(n_latents, obj)
        self.n_latents = n_latents
        self.vaes["mod_1"] = VAE(enc = "CNN", dec= "CNN", feature_dim=[3,64,64], n_latents = self.n_latents, ltype="bce")
        self.vaes["mod_2"] = VAE(enc = "CNN", dec= "CNN", feature_dim=[3,64,64], n_latents = self.n_latents, ltype="bce")

    def modality_mixing(self, mods):
        for key, mod in mods.items():
            pass
        return mods

def test_torch_mmvae_encode():
    bsize = 32
    mmvae = ExampleModel(n_latents=64, obj="elbo")
    inputs = {"mod_1": {"data":torch.rand((bsize, 3,64,64))}, "mod_2":{"data":torch.rand((bsize, 3,64,64))}}
    qz_xs = mmvae.encode(inputs)
    assert isinstance(qz_xs, dict)
    assert len(qz_xs.keys()) == len(mmvae.vaes.keys())
    for params in qz_xs.values():
        assert isinstance(params, dict)

def test_torch__mmvae_decode():
    bsize = 32
    mmvae = ExampleModel(n_latents=64, obj="elbo")
    qz_xs = {"mod_1": {"latents":torch.rand((bsize, mmvae.n_latents)).unsqueeze(0), "masks":None},
            "mod_2": {"latents":torch.rand((bsize, mmvae.n_latents)).unsqueeze(0), "masks":None}}
    px_zs = mmvae.decode(qz_xs)
    assert isinstance(px_zs, dict)
    assert len(px_zs.keys()) == len(mmvae.vaes.keys())
    for params in px_zs.values():
        assert isinstance(params, tuple)
        assert [d.shape == torch.Size((bsize,*[3,64,64])) for d in params]


def test_torch_mmvae_forward():
    bsize = 32
    mmvae = ExampleModel(n_latents=64, obj="elbo")
    inputs = {"mod_1": {"data": torch.rand((32, 3, 64, 64)), "masks": None}, "mod_2": {"data": torch.rand((32, 3, 64, 64))}, "masks": None}
    output = mmvae.forward(inputs)
    assert isinstance(output, VAEOutput)
    assert [isinstance(d, torch.distributions.Normal) for d in output.unpack_values()["decoder_dist"]]
    assert isinstance(output.unpack_values()["latent_samples"], list) and "latents" in output.unpack_values()["latent_samples"][0].keys()
    assert output.unpack_values()["latent_samples"][0]["latents"].shape == torch.Size([1, bsize, mmvae.n_latents])

