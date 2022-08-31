

def test_parse_args_file():
    from eval.infer import Config

    path = './data/config.yml'
    config = Config(path)
    assert config.epochs == 2

def test_parse_args_dir():
    from eval.infer import Config

    path = './data/'
    config = Config(path)
    assert config.epochs == 2
