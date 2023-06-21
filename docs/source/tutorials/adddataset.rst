.. _adddataset:

Add a new dataset
====================

By default, we support the proposed CdSprites+ dataset as well as MNIST, SVHN or the Caltech-UCSD Birds (CUB) dataset. Here we describe how
you can train the models on your own data.


Supported data formats, config
--------------------------------

In general, the preferred data formats (supported by default) are:

* pickle (``.pkl``)
* the pytorch format (``.pth``)
* a directory containing ``.png`` images

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

Eventhough the dataset is multimodal, a new instance of it will be created for each modality. Therefore,
the constructor gets two arguments: path to the modality (str) and modality_type (str). Modality type is any string
that you assign to the given modality to distinguish it from the others. For CUB we chose "image" for images and "text" for text, for MNIST_SVHN
we have "mnist" and "svhn". You specify mod_type in the config.

You also need to specify the expected shape of the data in the class attribute "feature_dims". This will be used by the dataset class to postprocess the data (i.e. reconstructions produced by the model),
 but also by the encoder and decoder networks to adjust sizes of the network layers.

Next thing you need are methods that prepare each modality for training (_preprocess_text and _preprocess_images). Here we use _preprocess_images from CdSprites+, since it is the same format, and only rewrite _preprocess_text.  Data loading is handled automatically by BaseDataset, so you
only perform reshaping, converting to tensors etc., so that these functions return tensors of the same length on the output.
Note: In case of sequential data (like text here), we make boolean masks and concatenate them with the last dimension of the text data. This is then automatically handled by the collate function.

The last thing we need to do is map the data processing functions to the modality types, i.e. define _mod_specific_loaders() and _mod_specific_savers():

.. code-block:: yaml

    def _mod_specific_loaders(self):
        return {"image": self._preprocess_images, "text": self._preprocess_text}

    def _mod_specific_savers(self):
        return {"image": self._postprocess_images, "text": self._postprocess_text}

Here we just assign the above-mentioned methods to the selected mod_types. Once this is done, the dataset class should be ready and you can launch training.

Different data formats
------------------------

If you want to train on an unsupported data format, you can file an issue on our `GitHub repository <https://github.com/gabinsane/multimodal-vae-comparison>`_.
Alternatively, you can try to incorporate it on your own. You will need to adjust two methods in the ``utils.py``.

First, add your new data format in ``get_path_type()`` so that it is recognised from the path.

.. code-block:: python

    def get_path_type(self, path):
        """
        See if the provided data path is supported.

        :param path: Path to the dataset
        :type path: str
        :return: recognised type of the data
        :rtype: str
        """
        assert os.path.exists(path), "Path does not exist: {}".format(path)
        if os.path.isdir(path):
            return "dir"
        if path[-4:] == ".pth":
            return "torch"
        if path[-4:] == ".pkl":
            return "pickle"
        raise Exception("Unrecognized dataset format. Supported types are: .pkl, .pth or directory with images")

Next, decide how you will load the data.

.. code-block:: python

    def load_data(self):
        """
        Loads the data from path

        :return: data prepared for training
        :rtype: torch.tensor
        """
        dtype = self.get_path_type(self.pth)
        if dtype == "dir":
            d = load_images(self.pth, self.data_dim)
        elif dtype == "torch":
            d = torch.load(self.pth)
        elif dtype == "pickle":
            with open(self.pth, 'rb') as handle:
                 d = pickle.load(handle)
        d = self.prepare_for_encoder(d)
        return d

Please note that by default, we have incorporated encoders and decoders for images (preferably in 32x32x3 or 64x64x3 resolution, resp. 28x28x1 pixels for MNIST),
text data (arbitrary strings which we encode on the character-level) and sequential data (e.g. actions suitable for a Transformer network). If you add a new data structure or image resolution,
you will also need to add new encoder and decoder networks - you can then specify these in the config file.
