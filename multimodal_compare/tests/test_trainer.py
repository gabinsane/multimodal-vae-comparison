import torch

from models.trainer import Trainer


def test_make_trainer():
    from eval.infer import parse_args
    path = './data/config.json'
    config, mods = parse_args(path)
    assert config['epochs'] == 2

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config, dev)

    assert isinstance(trainer.model, torch.nn.Module)

    check_optimizer(trainer, config)

    config['optimizer'] = 'adabelief'
    # trainer = Trainer(config, dev)
    # check_optimizer(trainer, config)


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
