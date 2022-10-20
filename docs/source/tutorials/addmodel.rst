.. _addmodel:

Add a new model
====================

We encourage the authors to implement their own multimodal VAE models into our toolkit. Here we describe how to do it.

General requirements
---------------------

The toolkit is written in PyTorch using the `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ framework and we expect new models to use this framework as well. Currently, it is
possible to implement unimodal VAEs and any multimodal VAEs which use dedicated VAE instances for each modality.
You can add a new objective, encoder/decoder networks and of course other support modules that are needed.

Below we show a step-by-step tutorial on how to add a new model.


Adding a new model
---------------------

First, we start by defining the model in ``mmvae_models.py``. Our model will need a name and should inherit the TorchMMVAE
class defined in ``mmvae_base.py``.
``self.modelName`` will be used for model selection from the config file.

.. code-block:: python

   class POE(TorchMMVAE):
       def __init__(self, vaes:list, n_latents:int, obj_config:dict, model_config=None):
           """
           Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public

           :param vaes: list of modality-specific vae objects
           :type vaes: list
           :param n_latents: dimensionality of the (shared) latent space
           :type n_latents: int
           :param obj_cofig: config with objective-specific parameters (obj name, beta.)
           :type obj_config: dict
           :param model_cofig: config with model-specific parameters
           :type model_config: dict
           """
           super().__init__(n_latents, **obj_config)
           self.vaes = nn.ModuleDict(vaes)
           self.model_config = model_config
           self.modelName = 'poe'
           self.pz = dist.Normal
           self.prior_dist = dist.Normal


The TorchMMVAE class includes the bare functional minimum for a multimodal VAE, i.e. the forward pass, encode and decode functions and modality_mixing function.
The newly added model can override these methods or keep them as they are and only add the modality_mixing method. Here we add the ``forward()`` pass and all methods necessary for the multimodal data integration. The first input parameter
will be the multimodal data specified in a config where the keys label the modalities and values contain the data (and possibly masks where applicable).

.. code-block:: python
   :emphasize-lines: 1

    def forward(self, inputs, K=1):
        """
        Forward pass that takes input data and outputs a dict with  posteriors, reconstructions and latent samples
        :param inputs: input data, a dict of modalities where missing modalities are replaced with None
        :type inputs: dict
        :param K: sample K samples from the posterior
        :type K: int
        :return: dict where keys are modalities and values are a named tuple
        :rtype: dict
        """
        mu, logvar, single_params = self.modality_mixing(inputs, K)
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        qz_d, px_d, z_d = {}, {}, {}
        for mod, vae in self.vaes.items():
            px_d[mod] = vae.px_z(*vae.dec({"latents": z, "masks": inputs[mod]["masks"]}))
        for key in inputs.keys():
            qz_d[key] = qz_x
            z_d[key] = {"latents": z, "masks": inputs[key]["masks"]}
        return self.make_output_dict(single_params, px_d, z_d, joint_dist=qz_d)

    def modality_mixing(self, x, K=1):
        """
        Inference module, calculates the joint posterior
        :param inputs: input data, a dict of modalities where missing modalities are replaced with None
        :type inputs: dict
        :param K: sample K samples from the posterior
        :type K: int
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        batch_size = find_out_batch_size(x)
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents), use_cuda=True)
        single_params = {}
        for m, vae in self.vaes.items():
            if x[m]["data"] is not None:
                mod_mu, mod_logvar = vae.enc(x[m])
                single_params[m] = dist.Normal(*[mod_mu, mod_logvar])
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)
        # product of experts to combine gaussians
        mu, logvar = super(POE, POE).product_of_experts(mu, logvar)
        return mu, logvar, single_params


    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if use_cuda:
            mu, logvar = mu.to("cuda"), logvar.to("cuda")
        return mu, logvar


The ``forward()`` method must return the VAEOutput object located in output_storage.py. Proper placement of the outputs inside this object is handled automatically by TorchMMVAE, you can thus call
``self.make_output_dict(encoder_dist=None, decoder_dist=None, latent_samples=None, joint_dist=None, enc_dist_private=None, dec_dist_private=None, joint_decoder_dist=None, cross_decoder_dist=None)``. All these arguments are optional
(depends on what your objective function will need) and must be dictionaries with modality names as keys (i.e. {"mod_1: data,, "mod_2": data2}).

Next, we need to specify the objective() function for this model which will define the training procedure.

.. code-block:: python
   :emphasize-lines: 13, 25, 26

    def objective(self, mods):
        """
        Objective function for PoE

        :param data: input data with modalities as keys
        :type data: dict
        :return obj: dictionary with the obligatory "loss" key on which the model is optimized, plus any other keys that you wish to log
        :rtype obj: dict
        """
        lpx_zs, klds, losses = [[] for _ in range(len(mods.keys()))], [], []
        mods_inputs = subsample_input_modalities(mods)
        for m, mods_input in enumerate(mods_inputs):
            output = self.forward(mods_input)
            output_dic = output.unpack_values()
            kld = self.obj_fn.calc_kld(output_dic["joint_dist"][0], self.pz(*self.pz_params.to("cuda")))
            klds.append(kld.sum(-1))
            loc_lpx_z = []
            for mod in output.mods.keys():
                px_z = output.mods[mod].decoder_dist
                self.obj_fn.set_ltype(self.vaes[mod].ltype)
                lpx_z = (self.obj_fn.recon_loss_fn(px_z, mods[mod]) * self.vaes[mod].llik_scaling).sum(-1)
                loc_lpx_z.append(lpx_z)
                if mod == "mod_{}".format(m + 1):
                    lpx_zs[m].append(lpx_z)
            d = {"lpx_z": torch.stack(loc_lpx_z).sum(0), "kld": kld.sum(-1), "qz_x": output_dic["encoder_dist"], "zs": output_dic["latent_samples"], "pz": self.pz, "pz_params": self.pz_params}
            losses.append(self.obj_fn.calculate_loss(d)["loss"])
        ind_losses = [-torch.stack(m).sum() / self.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs)]
        obj = {"loss": torch.stack(losses).sum(), "reconstruction_loss": ind_losses, "kld": torch.stack(klds).mean(0).sum()}
        return obj

In this case, we use the subsampling strategy. We retrieve outputs from the model (line 13), calculate reconstruction losses and KL-divergences. To calculate ELBO (or any other objective),
use ``self.obj_fn which`` is an instance of MultimodalObjective in objectives.py. It contains all reconstruction loss terms and objectives like ELBO or IWAE (more to be added). Using these functions helps
unifying the code parts that should be shared among models.

The ``objective()`` function must return a dictionary which includes the "loss" key and stores a 1D torch.tensor with the computed loss. This will be passed
to the optimizer. You can also add other arbitrary keys that will be automatically logged in tensorboard.

Finally, we need to add our model to the list of all models in ``__init__.py`` located in the ``models`` directory:

.. code-block:: python
   :emphasize-lines: 2, 6

    from .mmvae_models import MOE as moe
    from .mmvae_models import POE as poe
    from .mmvae_models import MoPOE as mopoe
    from .mmvae_models import DMVAE as dmvae
    from .vae import VAE
    __all__ = [dmvae, moe, poe, mopoe, VAE]

If we need to, we can define specific encoder and decoder networks (although we can also re-use the existing ones).

Now we should be able to train using this model. We need to create a ``config.yml`` file as follows:

.. code-block:: yaml
   :emphasize-lines: 7

   batch_size: 32
   epochs: 700
   exp_name: poe_exp
   labels: ./data/mnist_svhn/labels.pkl
   lr: 1e-3
   beta: 1.5
   mixing: poe
   n_latents: 10
   obj: elbo
   optimizer: adam
   pre_trained: null
   seed: 2
   viz_freq: 20
   test_split: 0.1
   dataset_name: mnist_svhn
   modality_1:
      decoder: MNIST
      encoder: MNIST
      mod_type: image
      recon_loss:  bce
      path: ./data/mnist_svhn/mnist
   modality_2:
      decoder: SVHN
      encoder: SVHN
      recon_loss:  bce
      mod_type: image
      path: ./data/mnist_svhn/svhn

You can see that we specified "poe" as our multimodal mixing model. After configuring the experiment, we can run the training:


.. code-block:: python

   cd multimodal-compare
   python main.py --cfg ./configs/config.yml

