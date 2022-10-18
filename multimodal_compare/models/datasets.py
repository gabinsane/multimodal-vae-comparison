import torch
import cv2
import numpy as np
import math, copy
from utils import one_hot_encode, output_onehot2text, lengths_to_mask, turn_text2image, load_data, add_recon_title
from torchvision.utils import make_grid


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

    def _postprocess_all2img(self, data):
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type == "text" else output_processed
        return output_processed

    def save_traversals(self, recons, path):
        output_processed = torch.tensor(self._postprocess_all2img(recons)).transpose(1, 3)
        grid = np.asarray(make_grid(output_processed, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0))
        cv2.imwrite(path, cv2.cvtColor(grid.astype("uint8"), cv2.COLOR_BGR2RGB))


# ----- Multimodal Datasets ---------

class GEBID(BaseDataset):
    feature_dims = {"image": [64, 64, 3],
                    "text": [52, 27, 1]
                    }

    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth
        self.text2img_size = (64,192,3)

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "text": self._postprocess_text}

    def _preprocess_images(self):
        return super(GEBID, self)._preprocess_images([self.feature_dims["image"][i] for i in [2,0,1]])

    def _postprocess_images(self, data):
        if isinstance(data, dict):
            data = data["data"]
        return np.asarray(data.detach().cpu())*255

    def _postprocess_text(self, data):
        if isinstance(data, dict):
            masks = data["masks"]
            data = data["data"]
            text = output_onehot2text(data)
            if masks is not None:
                masks = torch.count_nonzero(masks, dim=-1)
                text = [x[:masks[i]] for i, x in enumerate(text)]
            return text
        else:
            return output_onehot2text(data)

    def _preprocess_text(self):
        return self._preprocess_text_onehot()

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess_all2img(recons)
        outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = turn_text2image(output, img_size=self.text2img_size) if mod_names[key] == "text" \
                else np.reshape(output,(-1,*self.feature_dims["image"]))
            images = add_recon_title(images, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3))*125)
        inputs = np.hstack(input_processed).astype("uint8")
        final = np.hstack((inputs, np.vstack(outs).astype("uint8")))
        cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

class CUB(GEBID):
    feature_dims = {"image": [64, 64, 3],
                    "text": [246, 27, 1]
                    }

    def __init__(self, pth, mod_type):
        super().__init__(pth, mod_type)
        self.mod_type = mod_type
        self.path = pth
        self.text2img_size = (64,256,3)

    def _preprocess_text_onehot(self):
        """
        General function for loading text strings and preparing them as torch one-hot encodings

        :return: torch with text encodings and masks
        :rtype: torch.tensor
        """
        self.has_masks = True
        self.categorical = True
        data = [one_hot_encode(len(f), f) for f in self.get_data_raw()]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

    def _postprocess_text(self, data):
        if isinstance(data, dict):
            masks = data["masks"]
            data = data["data"]
            text = output_onehot2text(data)
            if masks is not None:
                masks = torch.count_nonzero(masks, dim=-1)
                text = [x[:masks[i]] for i, x in enumerate(text)]
        else:
            text = output_onehot2text(data)
        for i, phrase in enumerate(text):
            phr = phrase.split(" ")
            newphr = copy.deepcopy(phr)
            stringcount = 0
            for x, w in enumerate(phr):
                stringcount += (len(w))+1
                if stringcount > 40:
                    newphr.insert(x, "\n")
                    stringcount = 0
            text[i] = (" ".join(newphr)).replace("\n  ", "\n ")
        return text


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

    def _mod_specific_savers(self):
        return {"mnist": self._postprocess_mnist, "svhn": self._postprocess_svhn}

    def _postprocess_svhn(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.detach().cpu()).reshape(-1, *self.feature_dims["svhn"]) * 255
        images_res = []
        for i in images:
            images_res.append(cv2.resize(i, (28,28)))
        return np.asarray(images_res)

    def _postprocess_mnist(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.detach().cpu()).reshape(-1,*self.feature_dims["mnist"])*255
        images_3chan = cv2.merge((images, images, images)).squeeze(-2)
        return images_3chan

    def _process_mnist(self):
        return super(MNIST_SVHN, self)._preprocess_images([self.feature_dims["mnist"][i] for i in [2,0,1]])

    def _process_svhn(self):
        return super(MNIST_SVHN, self)._preprocess_images([self.feature_dims["svhn"][i] for i in [2,0,1]])

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess(recons)
        outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = add_recon_title(output, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3))*125)
        inputs = np.hstack(input_processed).astype("uint8")
        final = np.hstack((inputs, np.vstack(outs).astype("uint8")))
        cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
