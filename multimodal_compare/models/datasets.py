import torch
import numpy as np
from utils import get_mean, kl_divergence, load_images, lengths_to_mask, load_data
from utils import one_hot_encode, output_onehot2text, lengths_to_mask


class BaseDataset():
    def __init__(self, pth, mod_type):
        self.path = pth
        self.mod_type = mod_type
        self.has_masks = False
        self.categorical = False

    def _mod_specific_fns(self):
        raise NotImplementedError

    def _preprocess(self):
        assert self.mod_type in self._mod_specific_fns().keys(), "Unsupported modality type for {}".format(self.path)
        return self._mod_specific_fns()[self.mod_type]()

    def get_data_raw(self):
        data = load_data(self.path)
        return data

    def get_data(self):
        return self._preprocess()

#----- Multimodal Datasets ---------

class GEBID(BaseDataset):
    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth

    def _mod_specific_fns(self):
        return {"image": self._process_images, "text": self._process_text}

    def _process_images(self):
        data = [torch.from_numpy(np.asarray(x.reshape(3, 64,64)).astype(np.float)) for x in self.get_data_raw()]
        return torch.stack(data)

    def _process_text(self):
        self.has_masks = True
        self.categorical = True
        data = [" ".join(x) for x in self.get_data_raw()]
        data = [one_hot_encode(len(f), f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data])))
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        return data


class MNIST_SVHN(BaseDataset):
    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth

    def _mod_specific_fns(self):
        return {"mnist": self._process_mnist(), "svhn": self._process_svhn()}

    def _process_mnist(self):
        data = [torch.from_numpy(np.asarray(x.reshape(1, 28, 28)).astype(np.float)) for x in self.get_data_raw()]
        return torch.stack(data)

    def _process_svhn(self):
        data = [torch.from_numpy(np.asarray(x.reshape(3, 32, 32)).astype(np.float)) for x in self.get_data_raw()]
        return torch.stack(data)