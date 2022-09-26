import torch
import numpy as np
from utils import get_mean, kl_divergence, load_images, lengths_to_mask, load_data
from utils import one_hot_encode, output_onehot2text, lengths_to_mask


class BaseDataset():
    """
    Abstract dataset class shared for all datasets
    """

    def __init__(self, pth, mod_type):
        """

        :param pth: path to the given modality
        :type pth: str
        :param mod_type: tag for the modality for correct processing (e.g. "text", "image", "mnist", "svhn" etc.)
        :type mod_type: str
        """
        assert hasattr(self, "feature_dims"), "Dataset class must have the feature_dims attribute"
        self.path = pth
        self.mod_type = mod_type
        self.has_masks = False
        self.categorical = False

    def _mod_specific_loaders(self):
        """
        Assigns the preprocessing function based on the mod_type
        """
        raise NotImplementedError

    def _mod_specific_savers(self):
        """
        Assigns the postprocessing function based on the mod_type
        """
        raise NotImplementedError

    def _preprocess(self):
        """
        Preprocesses the loaded data according to modality type

        :return: preprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_loaders().keys(), "Unsupported modality type for {}".format(
            self.path)
        return self._mod_specific_loaders()[self.mod_type]()

    def _postprocess(self, output_data):
        """
        Postprocesses the output data according to modality type

        :return: postprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_savers().keys(), "Unsupported modality type for {}".format(self.path)
        return self._mod_specific_savers()[self.mod_type](output_data)

    def get_data_raw(self):
        """
        Loads raw data from path

        :return: loaded raw data
        :rtype: list
        """
        data = load_data(self.path)
        return data

    def get_data(self):
        """
        Returns processed data

        :return: processed data
        :rtype: list
        """
        return self._preprocess()

    def _preprocess_images(self, dimensions):
        """
        General function for loading images and preparing them as torch tensors

        :param dimensions: feature_dim for the image modality
        :type dimensions: list
        :return: preprocessed data
        :rtype: torch.tensor
        """
        data = [torch.from_numpy(np.asarray(x.reshape(*dimensions)).astype(np.float)) for x in self.get_data_raw()]
        return torch.stack(data)

    def _preprocess_text_onehot(self):
        """
        General function for loading text strings and preparing them as torch one-hot encodings

        :return: torch with text encodings and masks
        :rtype: torch.tensor
        """
        self.has_masks = True
        self.categorical = True
        data = [" ".join(x) for x in self.get_data_raw()]
        data = [one_hot_encode(len(f), f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks


# ----- Multimodal Datasets ---------

class GEBID(BaseDataset):
    feature_dims = {"image": [64, 64, 3],
                    "text": [52, 27, 1]
                    }

    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "text": self._postprocess_text}

    def _preprocess_images(self):
        return super(GEBID, self)._preprocess_images([self.feature_dims["image"][i] for i in [2,0,1]])

    def _postprocess_images(self, data):
        pass

    def _postprocess_text(self, data):
        pass

    def _preprocess_text(self):
        return self._preprocess_text_onehot()

    def save_recons(self, data, recons, path):
        output_processed = self._postprocess(recons)


class CUB(BaseDataset):
    feature_dims = {"image": [64, 64, 3],
                    "text": [52, 27, 1]
                    }

    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _process_images(self):
        return super(CUB, self)._preprocess_images([self.feature_dims["image"][i] for i in [2,0,1]])

    def _preprocess_text(self):
        return super(CUB)._preprocess_text_onehot()


class MNIST_SVHN(BaseDataset):
    feature_dims = {"mnist": [28,28,1],
                    "svhn": [32,32,3]
                    }

    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth

    def _mod_specific_loaders(self):
        return {"mnist": self._process_mnist, "svhn": self._process_svhn}

    def _process_mnist(self):
        return super(MNIST_SVHN, self)._preprocess_images([self.feature_dims["mnist"][i] for i in [2,0,1]])

    def _process_svhn(self):
        return super(MNIST_SVHN, self)._preprocess_images([self.feature_dims["svhn"][i] for i in [2,0,1]])
