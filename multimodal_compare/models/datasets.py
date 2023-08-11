import torch
import cv2, os
import numpy as np
import math, copy
import wget
from utils import one_hot_encode, output_onehot2text, lengths_to_mask, turn_text2image, load_data, add_recon_title
from torchvision.utils import make_grid
from eval.eval_sprites import eval_single_model as sprites_eval
from eval.eval_cdsprites import eval_single_model as cdsprites_eval
import imageio
import torchvision

class BaseDataset():
    """
    Abstract dataset class shared for all datasets
    """

    def __init__(self, pth, testpth, mod_type):
        """

        :param pth: path to the given modality
        :type pth: str
        :param mod_type: tag for the modality for correct processing (e.g. "text", "image", "mnist", "svhn" etc.)
        :type mod_type: str
        """
        assert hasattr(self, "feature_dims"), "Dataset class must have the feature_dims attribute"
        self.path = pth
        self.testdata = testpth
        self.current_path = None
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

    def labels(self):
        """Returns labels for the whole dataset"""
        return None

    def get_labels(self, split="train"):
        """Returns labels for the given split: train or test"""
        self.current_path = self.path if split == "train" else self.testdata
        return self.labels()

    def eval_statistics_fn(self):
        """(optional) Returns a dataset-specific function that runs systematic evaluation"""
        return None

    def current_datatype(self):
        """Returns whther the current path to data points to test data or train data"""
        if self.current_path == self.testdata:
            return "test"
        elif self.current_path == self.path:
            return "train"
        else:
            return None

    def _preprocess(self):
        """
        Preprocesses the loaded data according to modality type

        :return: preprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_loaders().keys(), "Unsupported modality type for {}".format(
            self.current_path)
        return self._mod_specific_loaders()[self.mod_type]()

    def _postprocess(self, output_data):
        """
        Postprocesses the output data according to modality type

        :return: postprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_savers().keys(), "Unsupported modality type for {}".format(self.current_path)
        return self._mod_specific_savers()[self.mod_type](output_data)

    def get_processed_recons(self, recons_raw):
        """
        Returns the postprocessed data that came from the decoders

        :param recons_raw: tensor with output reconstructions
        :type recons_raw: torch.tensor
        :return: postprocessed data as returned by the specific _postprocess function
        :rtype: list
        """
        return self._postprocess(recons_raw)

    def get_data_raw(self):
        """
        Loads raw data from path

        :return: loaded raw data
        :rtype: list
        """
        data = load_data(self.current_path)
        return data

    def get_data(self):
        """
        Returns processed data

        :return: processed data
        :rtype: list
        """
        self.current_path = self.path
        return self._preprocess()

    def get_test_data(self):
        """
        Returns processed test data if available

        :return: processed data
        :rtype: list
        """
        if self.testdata is not None:
            self.current_path = self.testdata
            return self._preprocess()
        return None

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
        data = []
        for x in self.get_data_raw():
            d = " ".join(x) if isinstance(x, list) else x
            data.append(d)
        data = [one_hot_encode(len(f), f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type not in ["text", "atts"] else output_processed
        return output_processed

    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if len(recons.shape) < 3:
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons))).transpose(1, 3)
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=num_dims))
            cv2.imwrite(path, cv2.cvtColor(np.transpose(grid, (1,2,0)).astype("uint8"), cv2.COLOR_BGR2RGB))
        else:
            output_processed = torch.stack([torch.tensor(np.array(self._postprocess_all2img(x.unsqueeze(0)))) for x in recons])
            output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
            rows = []
            for ind, dim in enumerate(output_processed):
                rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
            cv2.imwrite(path, cv2.cvtColor(np.vstack(np.asarray(rows)), cv2.COLOR_BGR2RGB))



# ----- Multimodal Datasets ---------

class CDSPRITESPLUS(BaseDataset):
    feature_dims = {"image": [64, 64, 3],
                    "text": [45, 27, 1]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.set_vis_image_shape()

    def set_vis_image_shape(self):
        width = 192
        if "level1" in self.path:
            width = 70
        elif "level2" in self.path:
            width = 120
        self.text2img_size = (64,width,3)

    def labels(self):
        """
        Extract text labels based on the dataset level
        :return: list of labels as strings
        :rtype: list
        """
        labels = [x.decode("utf8") for x in self.get_data_raw()["text"]]
        if "level2" in self.path:
            labels = [[x.split(" ")[0], x.split(" ")[1]] for x in labels]
        if "level3" in self.path:
            labels = [[x.split(" ")[0], x.split(" ")[1], x.split(" ")[2]] for x in labels]
        if "level4" in self.path:
            labels = [[x.split(" ")[0], x.split(" ")[1], x.split(" ")[2], " ".join(x.split(" ")[3:6])] for x in labels]
        if "level5" in self.path:
            labels = [[x.split(" ")[0], x.split(" ")[1], x.split(" ")[2], " ".join(x.split(" ")[3:6]),
                       " ".join(x.split(" ")[6:])] for x in labels]
        return labels

    def eval_statistics_fn(self):
        return cdsprites_eval

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "text": self._postprocess_text}

    def _preprocess_images(self):
        d = self.get_data_raw()["image"][:].reshape(-1, *[self.feature_dims["image"][i] for i in [2,0,1]])
        data = torch.tensor(d)/255
        return data

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
        d = self.get_data_raw()["text"]
        self.has_masks = True
        self.categorical = True
        data = [one_hot_encode(len(f), f.decode("utf8")) for f in d]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

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
        cv2.imwrite(path, final)

class CUB(BaseDataset):
    """Dataset class for our processed version of Caltech-UCSD birds dataset. We use the original images and text
    represented as sequences of one-hot-encodings for each character (incl. spaces)"""
    feature_dims = {"image": [64, 64, 3],
                    "text": [246, 27, 1]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.text2img_size = (64,380,3)

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

    def labels(self):
        """
        No labels for T-SNAE available
        """
        return None

    def _preprocess_text(self):
        d = self.get_data_raw()
        self.has_masks = True
        self.categorical = True
        data = [one_hot_encode(len(f), f) for f in d]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

    def _preprocess_images(self):
        d = self.get_data_raw().reshape(-1, *[self.feature_dims["image"][i] for i in [2,0,1]])
        data = torch.tensor(d)
        return data

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "text": self._postprocess_text}

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

class MNIST_SVHN(BaseDataset):
    """Dataset class for the MNIST-SVHN bimodal dataset (can be also used for unimodal training)"""
    feature_dims = {"mnist": [28,28,1],
                    "svhn": [32,32,3]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.text2img_size = (32, 32, 3)
        self.check_indices_present()

    def check_indices_present(self):
        if not os.path.exists(self.path):
            wget.download(os.path.join("https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/",
                                       os.path.basename(self.path)), out=os.path.dirname(self.path))

    def labels(self):
        return self.train_labels

    def _mod_specific_loaders(self):
        return {"mnist": self._process_mnist, "svhn": self._process_svhn}

    def _mod_specific_savers(self):
        return {"mnist": self._postprocess_mnist, "svhn": self._postprocess_svhn}

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        return output_processed

    def _postprocess_svhn(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.detach().cpu().reshape(-1, 3,32,32)).transpose(0,2,3,1) * 255
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
        data = torchvision.datasets.MNIST('../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        t_mnist = torch.load(self.path)[1::7][:200000]
        d = data.train_data[t_mnist].unsqueeze(1)
        self.train_labels = data.train_labels[t_mnist]
        return d /255

    def _process_svhn(self):
        data = torchvision.datasets.SVHN('../data', download=True, split='train', transform=torchvision.transforms.ToTensor())
        t_svhn = torch.load(self.path)[1::7][:200000]
        d = data.data[t_svhn]
        self.train_labels = data.labels[t_svhn]
        return (torch.tensor(d))/255

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

class SPRITES(BaseDataset):
    feature_dims = {"frames": [8,64,64,3],
                    "attributes": [4,6],
                    "actions": [9]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.text2img_size = (64, 145, 3)
        self.directions = ['front', 'left', 'right']
        self.actions = ['walk', 'spellcard', 'slash']
        self.label_map = ["walk front", "walk left", "walk right", "spellcard front", "spellcard left",
                          "spellcard right", "slash front", "slash left", "slash right"]
        self.attr_map = ["skin", "pants", "top", "hair"]
        self.att_names = [["pink", "yellow", "grey", "silver", "beige", "brown"], ["white", "gold", "red", "armor", "blue", "green"],
                          ["maroon", "blue", "white", "armor", "brown", "shirt"],["green", "blue", "yellow", "silver", "red", "purple"]]

    def labels(self):
        if self.current_path is None:
            return None
        actions = np.argmax(self.get_actions()[:, :9], axis=-1)
        labels = []
        for a in actions:
            labels.append(self.label_map[int(a)])
        return labels

    def eval_statistics_fn(self):
        return sprites_eval

    def get_frames(self):
        X_train = []
        for act in range(len(self.actions)):
            for i in range(len(self.directions)):
                x = np.load(os.path.join(self.current_path, '{}_{}_frames_{}.npy'.format(self.actions[act], self.directions[i], self.current_datatype())))
                X_train.append(x)
        data = np.concatenate(X_train, axis=0)
        return torch.tensor(data)

    def get_attributes(self):
        self.categorical = True
        A_train = []
        for act in range(len(self.actions)):
            for i in range(len(self.directions)):
                a = np.load(os.path.join(self.current_path, '{}_{}_attributes_{}.npy'.format(self.actions[act], self.directions[i], self.current_datatype())))
                A_train.append(a[:, 0, :, :])
        data = np.concatenate(A_train, axis=0)
        return torch.tensor(data)

    def get_actions(self):
        self.categorical = True
        D_train = []
        for act in range(len(self.actions)):
            for i in range(len(self.directions)):
                a = np.load(os.path.join(self.current_path, '{}_{}_attributes_{}.npy'.format(self.actions[act], self.directions[i], self.current_datatype())))
                d = np.zeros([a.shape[0], 9])
                d[:, 3 * act + i] = 1
                D_train.append(d)
        data = np.concatenate(D_train, axis=0)
        return torch.tensor(data)

    def make_masks(self, shape):
        return torch.ones(shape).unsqueeze(-1)

    def _mod_specific_loaders(self):
        return {"frames": self.get_frames, "attributes": self.get_attributes, "actions": self.get_actions}

    def _mod_specific_savers(self):
        return {"frames": self._postprocess_frames, "attributes": self._postprocess_attributes,
                "actions": self._postprocess_actions}

    def _postprocess_frames(self, data):
        data = data["data"] if isinstance(data, dict) else data
        return np.asarray(data.detach().cpu().reshape(-1, *self.feature_dims["frames"])) * 255

    def _postprocess_actions(self, data):
        data = data["data"] if isinstance(data, dict) else data
        indices = np.argmax(data.detach().cpu(), axis=-1)
        return [self.label_map[int(i)] for i in indices]

    def _postprocess_attributes(self, data):
        data = data["data"] if isinstance(data, dict) else data
        indices = np.argmax(data.detach().cpu(), axis=-1)
        atts = []
        for i in indices:
            label = ""
            for att_i, a in enumerate(i):
                label += self.att_names[att_i][a] + " " + self.attr_map[att_i]
                label += " \n" if att_i in [0,1,3] else ", "
            atts.append(label)
        return atts

    def iter_over_inputs(self, outs, data, mod_names, f=0):
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = turn_text2image(output, img_size=self.text2img_size) if mod_names[key] in ["attributes", "actions"] \
                else output[:, f, :, :, :]
            images = add_recon_title(images, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3)) * 145)
        inputs = np.hstack(input_processed).astype("uint8")
        return np.hstack((inputs, np.vstack(outs).astype("uint8")))

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess_all2img(recons)
        if self.mod_type != "frames" and [k for k, v in mod_names.items() if v == 'frames'][0] not in data.keys():
            outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
            final = self.iter_over_inputs(outs, data, mod_names)
            cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        else:
            timesteps = []
            for f in range(8):
                outs = add_recon_title(output_processed[:, f, :, :, :], "output\n{}".format(self.mod_type), (0, 170, 0))\
                if self.mod_type == "frames" else add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
                final = self.iter_over_inputs(outs, data, mod_names, f)
                timesteps.append(final)
            imageio.mimsave(path.replace(".png", ".gif"), timesteps)

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type in ["actions", "attributes"] else output_processed
        return output_processed

    def save_traversals(self, recons, path):
        """
        Makes a grid of traversals and saves as animated gif image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        """
        if self.mod_type != "frames":
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons))).transpose(1, 3)
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0))
            cv2.imwrite(path, cv2.cvtColor(grid.astype("uint8"), cv2.COLOR_BGR2RGB))
        else:
            grids = []
            output_processed = torch.tensor(self._postprocess_all2img(recons)).permute(1,0,4,3,2)
            for i in output_processed:
                grids.append(np.asarray(make_grid(i, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0)).astype("uint8"))
            imageio.mimsave(path.replace(".png", ".gif"), grids)

class CELEBA(BaseDataset):
    feature_dims = {"image": [64, 64, 3],
                    "atts": [4, 2],
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.text2img_size = (64,192,3)
        self.labelmap = [["hairy", "bald"], ["no eyeglasses", "eyeglasses"], ["female", "male"], ["not smiling", "smiling"]]

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "atts": self._preprocess_atts}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "atts": self._postprocess_atts}

    def _preprocess_images(self):
        return super(CELEBA, self)._preprocess_images([self.feature_dims["image"][i] for i in [2,0,1]])

    def _postprocess_images(self, data):
        data = data["data"] if isinstance(data, dict) else data
        return np.asarray(data.detach().cpu())*255

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type in ["atts"] else output_processed
        return output_processed

    def _postprocess_atts(self, data):
        if isinstance(data, dict):
            data = data["data"]
        data = np.asarray([np.asarray([(s[0]) for s in x]) for x in data.detach().cpu()])
        data_str = []
        for s in data:
            d = []
            for idx, i in enumerate(s):
                d.append(self.labelmap[idx][int(i)])
            data_str.append(", ".join(d))
        return np.asarray(data_str)

    def _preprocess_atts(self):
        d = (torch.tensor(self.get_data_raw().astype("float32"))+1)/2
        data = []
        for s in d:
            sample = []
            for v in s:
                i = torch.tensor([0,1]) if v == 0 else torch.tensor([1,0])
                sample.append(i)
            data.append(torch.stack(sample))
        return torch.stack(data)

    def save_recons(self, data, recons, path, mod_names):
        output_processed = self._postprocess_all2img(recons)
        outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
        input_processed = []
        for key, d in data.items():
            output = self._mod_specific_savers()[mod_names[key]](d)
            images = turn_text2image(output, img_size=self.text2img_size) if mod_names[key] == "atts" \
                else np.reshape(output,(-1,*self.feature_dims["image"]))
            images = add_recon_title(images, "input\n{}".format(mod_names[key]), (0, 0, 255))
            input_processed.append(np.vstack(images))
            input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3))*125)
        inputs = np.hstack(input_processed).astype("uint8")
        final = np.hstack((inputs, np.vstack(outs).astype("uint8")))
        cv2.imwrite(path, final)

    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if len(recons.shape) < 5:
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons))).transpose(1, 3)
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0))
            cv2.imwrite(path, grid.astype("uint8"))
        else:
            output_processed = torch.stack([torch.tensor(self._postprocess_all2img(x)) for x in recons])
            output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
            rows = []
            for ind, dim in enumerate(output_processed):
                rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
            cv2.imwrite(path, np.vstack(np.asarray(rows)))

class FASHIONMNIST(BaseDataset):
    """Dataset class for the FashionMNIST dataset"""
    feature_dims = {"image": [28,28,1],
                    "label": [10],
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.labels_train = None

    def labels(self):
        return self.labels_train

    def get_data_raw(self):
        data = torchvision.datasets.FashionMNIST(root=self.path, train=True, download=True)
        self.labels_train = [int(x) for x in data.targets]
        return data.data.unsqueeze(-1)/255

    def _mod_specific_loaders(self):
        return {"image": self._process_image, "label": self._process_label}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_image, "label": self._postprocess_label}

    def _process_label(self, data):
        self.get_data_raw()
        d = np.zeros((self.labels_train.size, self.labels_train.max() + 1))
        d[np.arange(self.labels_train.size), self.labels_train] = 1
        return torch.tensor(d)

    def _postprocess_label(self, data):
        pass

    def _postprocess_image(self, data):
        data = data["data"] if isinstance(data, dict) else data
        images = np.asarray(data.detach().cpu()).reshape(-1,*self.feature_dims["image"])*255
        images_3chan = cv2.merge((images, images, images)).squeeze(-2)
        return images_3chan

    def _process_image(self):
        return super(FASHIONMNIST, self)._preprocess_images([self.feature_dims["image"][i] for i in [2,0,1]])

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

class POLYMNIST(BaseDataset):
    """Dataset class for the POLYMNIST dataset"""
    feature_dims = {"m0": [28,28,3],
                    "m1": [28, 28, 3],
                    "m2": [28, 28, 3],
                    "m3": [28, 28, 3],
                    "m4": [28, 28, 3],
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type

    def _mod_specific_loaders(self):
        d = {}
        for k in ["m0", "m1", "m2", "m3", "m4"]:
            d[k] = self._process_mnist
        return d

    def _mod_specific_savers(self):
        d = {}
        for k in ["m0", "m1", "m2", "m3", "m4"]:
            d[k] = self._postprocess_mnist
        return d

    def _postprocess_mnist(self, data):
        if isinstance(data, dict):
            data = data["data"]
        images = np.asarray(data.view(-1, 3,28,28).detach().cpu().permute(0,2,3,1))*255
        return images

    def _process_mnist(self):
        return self.get_data_raw()

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

    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if len(recons.shape) < 5:
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons)))
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=num_dims))
            cv2.imwrite(path, cv2.cvtColor(np.transpose(grid, (1,2,0)).astype("uint8"), cv2.COLOR_BGR2RGB))
        else:
            output_processed = torch.stack([torch.tensor(np.array(self._postprocess_all2img(x))) for x in recons])
            output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
            rows = []
            for ind, dim in enumerate(output_processed):
                rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
            cv2.imwrite(path, cv2.cvtColor(np.vstack(np.asarray(rows)), cv2.COLOR_BGR2RGB))