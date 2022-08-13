.. _adddataset:

Add a new dataset
====================

By default, we support the proposed GeBiD dataset as well as MNIST, SVHN or the Caltech-UCSD Birds (CUB) dataset. Here we describe how
you can train the models on your own data.


Supported data formats
-----------------------

By default, we have incorporated encoders and decoders for images (preferably in 32x32x3 or 64x64x3 resolution, resp. 28x28x1 pixels for MNIST),
text data (arbitrary strings which we encode on the character-level) and sequential data (e.g. actions suitable for a Transformer network).

The prefered data formats (supported by default) ar pickle (``.pkl``), the pytorch format (``.pth``) or a directory containing ``.png`` images.
To train with any of these, specify the path to your data in the ``config.yml``:


.. code-block:: yaml
   :emphasize-lines: 16,17,18, 19, 20, 22, 23, 24, 25, 26

    batch_size: 32
    epochs: 600
    exp_name: cub
    labels:
    loss: bce
    lr: 1e-3
    mixing: moe
    n_latents: 16
    obj: elbo
    optimizer: adam
    pre_trained: null
    seed: 2
    viz_freq: 10
    test_split: 0.1
    modality_1:
      path: ./data/cub/images
      feature_dim: [64, 64, 3]
      mod_type: image
      decoder: CNN
      encoder: CNN
    modality_2:
      path: ./data/cub/cub_captions.pkl
      feature_dim: [256,27,1]
      mod_type: text
      decoder: TxtTransformer
      encoder: TxtTransformer


This is an example of the config file for the CUB dataset (for download, see our
`README <https://github.com/gabinsane/multimodal-vae-comparison#training-on-other-datasets>`_).

As you can see, we specified the path to an image folder (``./data/cub/images``) and to the pickled captions (``./data/cub/cub_captions.pkl``). Both
modalities are expected to be ordered so that they can be semantically matched into pairs (e.g. the first image should match with the first caption).
We also provided the data feature dimensions (``feature_dim``) - this is helpful for visualization methods and for the encoder/decoder networks to
reshape the data as needed. In case of sequential data (text in this case), the first value should be the maximum length of a sequence in the dataset. The value
27 here corresponds to the one-hot encodings with the length of the alphabet.
``mod_type`` is a string which helps the visualization methods to recognize how to display the reconstructions. We currently only support "image" or "text".

Finally, specify the corresponding encoder and decoder networks which suit your data type.

If your own dataset is in the supported format, you should be ready to train just after making your config. If you find bugs,
please let us know.


Adding a new dataset
---------------------

If you want to train on an unsupported data format, you can file an issue on our `GitHub repository <https://github.com/gabinsane/multimodal-vae-comparison>`_.
Alternatively, you can try to incorporate it on your own. You will need to adjust three methods in the ``VaeDataset`` class in  ``vae.py``.

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

Finally, add any preprocessing that you need so that your data is in the ``torch.tensor`` format.

.. code-block:: python

    def prepare_for_encoder(self, data):
        """
        Prepares the data for training.

        :param data: the loaded data
        :type data: Union[list, torch.tensor, ndarray]
        :return: data reshaped for training,
        :rtype: torch.tensor
        """
        if self.network_type.lower() in ["transformer", "cnn", "3dcnn", "fnn"]:
            data = [torch.from_numpy(np.asarray(x).astype(np.float)) for x in data]
            if self.network_type in ["cnn", "fnn"]:
                data = torch.stack(data).transpose(1,3)
            if "transformer" in self.network_type.lower():
                if len(data[0].shape) < 3:
                    data = [torch.unsqueeze(i, dim=1) for i in data]
        elif "text" in self.mod_type:
            if len(data[0]) > 1 and not isinstance(data[0], str):
                data = [" ".join(x) for x in data] if not "cub_" in self.pth else data
            data = [one_hot_encode(len(f), f) for f in data]
            data = [torch.from_numpy(np.asarray(x)) for x in data]
            if "transformer" not in self.network_type.lower():
                data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        if self.network_type.lower() == "audioconv":
            self.prepare_audio(data)
        if "image" in self.mod_type:
            data = self.check_img_normalize(data)
        return data


That should be it. If needed, you can also add visualization methods to see the results during training. For unimodal VAE,
this would be the ``reconstruct()`` method in ``vae.py``. For multimodal VAEs, it is ``process_reconstructions()`` in ``mmvae_models.py``