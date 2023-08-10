import os
import pickle
import yaml
import argparse
from utils import get_root_folder, load_data


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
        self.K = 1
        self.model_cfg = {}
        self.mPath = None
        self.labels_test = None
        self.parser = parser
        self.eval_only = eval_only
        self.mods = []
        self.params = self.parse_params(parser)
        self._define_params()
        self._setup_savedir()

    def parse_params(self, parser):
        """
        Get parsed params

        :param parser: argument parser or str path to config
        :type parser: (argparse.ArgumentParser, str)
        :return: parsed parameters
        :rtype: dict
        """
        assert (isinstance(parser, argparse.ArgumentParser) or isinstance(parser, str) or isinstance(parser, dict))
        if isinstance(parser, argparse.ArgumentParser):
            self.params = self._parse_args()
        elif isinstance(parser, dict):
            self.params = parser
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
        return self.params

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

    def change_seed(self, seednum):
        self.seed = seednum
        self.params["seed"] = seednum

    def _get_mods_config(self, config):
        """
        Makes a list of all modality-specific dicts (self.modality_1, ..., self.modality_n), loads labels if provided
        """
        mods = sorted([x for x in dir(self) if "modality" in x])
        for m in mods:
            d = getattr(self, m)
            if not "private_latents" in d.keys():
                d["private_latents"] = None
            if not "llik_scaling" in d.keys():
                d["llik_scaling"] = 1
            self.mods.append(d)
        if config["labels"]:
            self.labels = load_data(config["labels"])

    def find_version(self):
        version = 0
        while True:
            versiondir = os.path.join(self.mPath, "version_{}".format(version))
            if os.path.exists(versiondir):
                version += 1
            else:
                return version

    def _setup_savedir(self):
        """
        Creates the model directory in the results folder and saves the config copy
        """
        self.mPath = os.path.join('results/', self.exp_name)
        version = self.find_version()
        self.mPath = os.path.join("results/", self.exp_name, "version_{}".format(version))
        if not self.eval_only:
            os.makedirs(self.mPath, exist_ok=True)
            os.makedirs(os.path.join(self.mPath, "visuals"), exist_ok=True)
            print('Experiment path:', self.mPath)
            self.dump_config()

    def dump_config(self):
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
        if isinstance(self.parser, argparse.ArgumentParser):
            args = self.parser.parse_args()
            config = self._load_config(args.cfg)
        else:
            config = self._load_config(self.parser)
        for name, value in vars(args).items():
            if value is not None and name != "cfg" and name in config.keys():
                config[name] = value
        return config
