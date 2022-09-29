from utils import get_torch_mean


def test_get_torch_mean():
    import torch
    l = [torch.tensor([i]) for i in [1.0, 2.0, 3.0]]
    res = get_torch_mean(l)

    assert res == 2.0
