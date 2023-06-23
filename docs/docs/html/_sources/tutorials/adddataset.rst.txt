.. _adddataset: center

Add a new dataset
====================

By default, we support the proposed CdSprites+ dataset as well as MNIST-SVHN, CelebA, SPRITES, PolyMNIST, FashionMNIST or the Caltech-UCSD Birds (CUB) dataset. Here we describe how
you can train the models on your own data.


Supported data formats, config
--------------------------------

In general, the preferred data formats (supported by default) are:

* pickle (``.pkl``)
* the pytorch format (``.pth``, ``.pt``)
* numpy format (``.npy``)
* hdf5 format (``.h5``)
* a directory containing ``.png`` or ``.jpg`` images

To train with any of these, specify the path to your data in the ``config.yml``:


.. code-block:: yaml
   :emphasize-lines: 15, 17,18, 19, 20, 21, 23, 24, 25, 26, 27

   batch_size: 16
   epochs: 600
   exp_name: cub
   labels:
   beta: 1
   lr: 1e-3
   mixing: moe
   n_latents: 16
   obj: elbo
   optimizer: adam
   pre_trained: null
   seed: 2
   viz_freq: 1
   test_split: 0.1
   dataset_name: cub
   modality_1:
     decoder: CNN
     encoder: CNN
     mod_type: image
     recon_loss:  bce
     path: ./data/cub/images
   modality_2:
     decoder: TxtTransformer
     encoder: TxtTransformer
     mod_type: text
     recon_loss: category_ce
     path: ./data/cub/cub_captions.pkl


This is an example of the config file for the CUB dataset (for download, see our
`README <https://github.com/gabinsane/multimodal-vae-comparison#training-on-other-datasets>`_).

As you can see, we specified the path to an image folder (``./data/cub/images``) and to the pickled captions (``./data/cub/cub_captions.pkl``). Both
modalities are expected to be ordered so that they can be semantically matched into pairs (e.g. the first image should match with the first caption).



Adding a new dataset class
---------------------------

If you wish to train on your own data, you will need to make a custom dataset class in ``datasets.py``. Any new dataset must inherit
from BaseDataset to have some common methods used by the DataModule.

In case of CUB we add it in datasets.py like this:

.. code-block:: python
   :linenos:

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

Eventhough the dataset is multimodal, a new instance of it will be created for each modality. Therefore,
the constructor gets two arguments: path to the modality (str) and eventually path to the test data (this is used for evaluation after training), and modality_type (str). Modality type is any string
that you assign to the given modality to distinguish it from the others. For CUB we chose "image" for images and "text" for text, for MNIST_SVHN
we have "mnist" and "svhn". You specify mod_type in the config.
You also need to specify the expected shape of the data in the class attribute "feature_dims". This will be used by the dataset class to postprocess the data (i.e. reconstructions produced by the model), but also by the encoder and decoder networks to adjust sizes of the network layers.

Next thing you need are methods that prepare each modality for training (``_preprocess_text`` and ``_preprocess_images``). Here we use ``_preprocess_images`` from CdSprites+, since it is the same format, and only rewrite _preprocess_text.  Data loading is handled automatically by BaseDataset, so you
only perform reshaping, converting to tensors etc., so that these functions return tensors of the same length on the output.
Note: In case of sequential data (like text here), we make boolean masks and concatenate them with the last dimension of the text data. This is then automatically handled by the collate function.

Another thing we need to do is map the data processing functions to the modality types, i.e. define ``_mod_specific_loaders()`` and ``_mod_specific_savers()``:

.. code-block:: yaml

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "text": self._postprocess_text}

Here we just assign the above-mentioned methods to the selected mod_types. Once this is done, the dataset class should be ready and you can launch training.

Finally, we can configure how are the outputs saved for visualization. This can be data-dependent, the ``save_recons()`` method shown in the example is suited
for putting images and text next to each other in one image. The ``_postprocess_all2img()`` method prints the string into image of the size ``self.text2image_size``
(defined in __init__, see Line 11).

Different data formats
------------------------

If you want to train on an unsupported data format, you can file an issue on our `GitHub repository <https://github.com/gabinsane/multimodal-vae-comparison>`_.
Alternatively, you can try to incorporate it on your own as it is only a matter of adjusting one function in ``utils.py``:

.. code-block:: python

    def load_data(path):
        """
        Returns loaded data based on path suffix
        :param path: Path to data
        :type path: str
        :return: loaded data
        :rtype: object
        """
        if path.startswith('.'):
            path = os.path.join(get_root_folder(), path)
        assert os.path.exists(path), "Path does not exist: {}".format(path)
        if os.path.isdir(path):
            return load_images(path)
        if pathlib.Path(path).suffix in [".pt",".pth"]:
            return torch.load(path)
        if pathlib.Path(path).suffix == ".pkl":
            return load_pickle(path)
        if pathlib.Path(path).suffix == ".h5":
            return h5py.File(path, 'r')
        if pathlib.Path(path).suffix == ".npy":
            return np.load(path)
        raise Exception("Unrecognized dataset format. Supported types are: .pkl, .pth or directory with images")


Please note that by default, we have incorporated encoders and decoders for images (preferably in 32x32x3 or 64x64x3 resolution, resp. 28x28x1 pixels for MNIST),
text data (arbitrary strings which we encode on the character-level) and sequential data (e.g. actions suitable for a Transformer network). If you add a new data structure or image resolution,
you will also need to add or adjust the encoder and decoder networks - you can then specify these in the config file.
