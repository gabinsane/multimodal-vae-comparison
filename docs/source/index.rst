Multimodal VAE Comparison Toolkit Documentation
================================================

The purpose of the Multimodal VAE Comparison toolkit is to offer a systematic and unified way to train, evaluate and compare the state-of-the-art
multimodal variational autoencoders. The toolkit can be used with arbitrary datasets and both uni/multimodal settings.
By default, we provide implementations of the Link text `MVAE <https://github.com/mhw32/multimodal-vae-public>`_
(`paper <https://arxiv.org/abs/1802.05335>`_), `MMVAE <https://github.com/iffsid/mmvae>`_
(`paper <https://arxiv.org/pdf/1911.03393.pdf>`_) and `MoPoE <https://github.com/thomassutter/MoPoE>`_
(`paper <https://openreview.net/forum?id=5Y21V0RDBV>`_) models, but anyone is free to contribute with their own
implementation.

We also provide a custom synthetic bimodal dataset, called GeBiD, designed specifically for comparison of the
joint- and cross-generative capabilities of multimodal VAEs. You can read about the utilities of the dataset in the proposed
paper (link will be added soon). This dataset offers 5 levels of difficulty (based on the number of attributes)
to find the minimal functioning scenario for each model. Moreover, its rigid structure enables automatic qualitative
evaluation of the generated samples. For more info, see below.

.. note::
   This page is currently a work in progress

-------------------

**Sub-Modules:**

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/addmodel
   tutorials/adddataset

.. toctree::
   :maxdepth: 1
   :caption: Code documentation

   code/trainer
