import os
import pickle
import yaml
import argparse
from utils import get_root_folder


class Config():
    """
    Config manager
    """

    def __init__(self, parser, eval_only=False):
        """
        :param parser: argument parser or str path to config
        :type parser: (argparse.ArgumentParser, str)
        """
        self.num_mods = None
        self.model_cfg = {}
        self.mPath = None
        self.parser = parser
        self.eval_only = eval_only
        self.mods = []
        assert (isinstance(parser, argparse.ArgumentParser) or isinstance(parser, str))
        if isinstance(parser, argparse.ArgumentParser):
            self.params = self._parse_args()
        elif isinstance(parser, str) and os.path.isfile(parser):
            self.params = self._load_config(parser)
        elif isinstance(parser, str) and os.path.isdir(parser):
            if os.path.isfile(os.path.join(parser, 'config.yml')):
                self.params = self._load_config(os.path.join(parser, 'config.yml'))
        elif os.path.isfile(os.path.join(get_root_folder(), parser)):
            self.params = self._load_config(os.path.join(get_root_folder(), parser))
        elif os.path.isfile(os.path.join(get_root_folder(), parser, 'config.yml')):
            self.params = self._load_config(os.path.join(get_root_folder(), parser, 'config.yml'))
        else:
            raise ValueError(f"{parser} is not a valid path nor parser")
        self._define_params()
        self._setup_savedir()

    def get_vis_dir(self):
        """
        :return: returns path to the model's visualisation directory
        :rtype: str
        """
        return os.path.join(self.mPath, "visuals/")

    def _define_params(self):
        """
        Sets up variables from config and retrieves modality-specific info
        """
        for p in self.params.keys():
            setattr(self, p, self.params[p])
        self._get_mods_config(self.params)

    def _get_mods_config(self, config):
        """
        Makes a list of all modality-specific dicts (self.modality_1, ..., self.modality_n), loads labels if provided
        """
        mods = sorted([x for x in dir(self) if "modality" in x])
        for m in mods:
            self.mods.append(getattr(self, m))
        if config["labels"]:
            with open(config["labels"], 'rb') as handle:
                self.labels = pickle.load(handle)

    def _setup_savedir(self):
        """
        Creates the model directory in the results folder and saves the config copy
        """
        self.mPath = os.path.join('results/', self.exp_name)
        if not self.eval_only:
            os.makedirs(self.mPath, exist_ok=True)
            os.makedirs(os.path.join(self.mPath, "visuals"), exist_ok=True)
            print('Experiment path:', self.mPath)
            with open('{}/config.yml'.format(self.mPath), 'w') as yaml_file:
                yaml.dump(self.params, yaml_file, default_flow_style=False)

    def _load_config(self, pth):
        with open(pth) as file:
            config = yaml.safe_load(file)
        return config

    def _parse_args(self):
        """
        Loads the .yml config specified in the --cfg argument. Any additional arguments override the values in the config.
        :return: dict; config
        :rtype: dict
        """
        args = self.parser.parse_args()
        config = self._load_config(args.cfg)
        for name, value in vars(args).items():
            if value is not None and name != "cfg" and name in config.keys():
                config[name] = value
        return config
