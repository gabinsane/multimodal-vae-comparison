import inspect
import torch

fields = ["encoder_dist", "joint_dist", "joint_decoder_dist", "decoder_dist", "dec_dist_private","latent_samples",
          "enc_dist_private", "cross_decoder_dist"]


class ModalityOutput:
    """
    Class that stores outputs for given modality
    """
    def __init__(self, id:str):
        """
        :param id: name of the modality (e.g. mod_1)
        :type id: str
        """
        self.id = id
        self.encoder_dist = None
        self.joint_dist = None
        self.decoder_dist = None
        self.dec_dist_private = None
        self.latent_samples = None
        self.enc_dist_private = None
        self.joint_decoder_dist = None
        self.cross_decoder_dist = None

    def set_value(self, field:str, val):
        """
        Sets value to the attribute

        :param field: name of the attribute (see line 5)
        :type field: str
        :param val: value to assign
        """
        if val is not None:
            self.check_field_valid(field)
            if field not in ["latent_samples", "cross_decoder_dist"]:
                self.check_is_distribution(val, field)
            else:
                assert isinstance(val, dict), "Expected {} to be a dict! Got {}".format(field, val)
        setattr(self, field, val)

    def get_value(self, field:str):
        """Returns the class attribute based on its name"""
        self.check_field_valid(field)
        return getattr(self, field)

    @staticmethod
    def check_is_distribution(val, field):
        """Checks if the val parameter is a torch.distribution instance"""
        assert isinstance(val, tuple([x[1] for x in inspect.getmembers(torch.distributions, inspect.isclass)])), \
            "{} value must be an instance of torch.distributions! Got: {}".format(field, val)

    @staticmethod
    def check_field_valid(field:str):
        """Checks if the input field belongs to supported attributes (see line 5)"""
        assert field in fields, "Unsupported field name {}. Choose out of: {}".format(field, fields)


class VAEOutput:
    """
    VAEOutput class for storing all kinds of VAE outputs
    """
    def __init__(self):
        self.mods = {}

    def add_new_modality(self, name:str):
        """Creates new instance of ModalityOutput with the provided id"""
        self.mods[name] = ModalityOutput(name)

    def set_value(self, mod:str, field:str, val):
        """
        Assigns the value val to the input field for the chosen modality mod
        :param mod: modality tag (e.g. "mod_1")
        :type mod: str
        :param field: field, a string from the supported fields (see line 5)
        :type field: str
        :param val: value to assign, differs based on the field
        :type val: torch.tensor
        """
        if mod not in self.mods.keys():
            self.add_new_modality(mod)
        self.mods[mod].set_value(field, val)

    def set_with_dict(self, d:dict, field:str):
        """
        Assigns values to modalities using a dict with modalities as keys

        :param d: input dictionary with modalities as keys
        :type d: dict
        :param field: name of the field to assign the values to (see line 5)
        :type field: str
        """
        if d is not None:
            for key in d.keys():
                self.set_value(key, field, d[key])

    def set_to_all(self, field:str, val):
        """
        Assigns the input value to the input field in all modalities
        :param field: field, a string from the supported fields (see line 3)
        :type field: str
        :param val: value to assign, differs based on the field
        :type val: torch.tensor
        """
        for key in self.mods.keys():
            self.set_value(key, field, val)

    def get_all_values(self, field):
        """Returns the values for the given field from all modalities"""
        vals = []
        for m in self.mods.values():
            vals.append(m.get_value(field))
        return vals

    def unpack_values(self):
        """
        Returns a dictionary where keys are fields (see line 5) and values are lists of values from all modalities

        :return: Unpacked data according to field type
        :rtype: dict
        """
        unpacked = {}
        for f in fields:
            unpacked[f] = self.get_all_values(f)
        return unpacked


