

def test_parse_args():
    from eval.infer import parse_args

    path = './data/config.json'
    config, mods = parse_args(path)
    assert config['epochs'] == 2

    path = './data/'
    config, mods = parse_args(path)
    assert config['epochs'] == 2