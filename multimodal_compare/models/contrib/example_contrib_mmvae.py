from ..mmvae_base import TorchMMVAE


class ExampleTorchVAE(TorchMMVAE):
    def __init__(self):
        super(ExampleTorchVAE).__init__()

    def forward(self, inputs, K=1):
        pass
