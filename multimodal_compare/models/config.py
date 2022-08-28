import os
import pickle
import yaml

class Config():
    def __init__(self, parser):
        """
        Config manager

        :param parser: argument parser
        :type parser: argparse.ArgumentParser
        """
        self.num_mods = None
        self.mPath = None
        self.parser = parser
        self.mods = []
        self.params = self._parse_args()
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
        Retrieves the modality-specific configs from the .yml config
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
        os.makedirs(self.mPath, exist_ok=True)
        os.makedirs(os.path.join(self.mPath, "visuals"), exist_ok=True)
        print('Experiment path:', self.mPath)
        with open('{}/config.yml'.format(self.mPath), 'w') as yaml_file:
            yaml.dump(self.params, yaml_file, default_flow_style=False)


    def _parse_args(self):
        """
        Loads the .yml config specified in the --cfg argument. Any additional arguments override the values in the config.
        :return: dict; config
        :rtype: dict
        """
        args = self.parser.parse_args()
        with open(args.cfg) as file:
            config = yaml.safe_load(file)
        for name, value in vars(args).items():
            if value is not None and name != "cfg" and name in config.keys():
                config[name] = value
        return config