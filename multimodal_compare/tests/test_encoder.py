import pytest
import torch

from models.encoders import VaeEncoder, Enc_CNN

def test_VaeEncoder_NotImplementedError():
    encoder = VaeEncoder(10, (10, 10, 3))
    x = torch.rand((encoder.data_dim))

    with pytest.raises(NotImplementedError):
        res = encoder.forward(x)
    # assert res.shape == encoder.latent_dim

def test_VaeEncoder_dims():
    batch_size = 32
    encoder = Enc_CNN(10, (3, 64, 64))
    assert issubclass(Enc_CNN, VaeEncoder), f"{encoder.__class__} is not a subclass of {VaeEncoder}"
    x = torch.rand((batch_size, *encoder.data_dim))

    mu, logvar = encoder.forward(x)

    assert mu.shape == (batch_size, encoder.latent_dim)