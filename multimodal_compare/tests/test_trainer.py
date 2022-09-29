import torch

from models.trainer import MultimodalVAE
from models.dataloader import DataModule

def test_make_trainer():
    from eval.infer import Config
    path = 'tests/data/config.yml'
    config = Config(path)
    assert config.epochs == 2

    data_module = DataModule(config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MultimodalVAE(config, data_module.get_dataset_class().feature_dims)

    assert isinstance(trainer.model, torch.nn.Module)




def check_optimizer(trainer, config):
    if config['optimizer'] == 'adam':
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    elif config['optimizer'] == 'adabelief':
        assert isinstance(trainer.optimizer, AdaBelief)

def test_adabelief():
    try:
        from adabelief_pytorch import AdaBelief
    except ModuleNotFoundError as e:
        assert False, f"AdaBelief is not available. {e}"
