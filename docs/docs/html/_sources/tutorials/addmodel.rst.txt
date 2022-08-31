.. _addmodel:

Add a new model
====================

We encourage the authors to implement their own multimodal VAE models into our toolkit. Here we describe how to do it.

General requirements
---------------------

The toolkit is written using PyTorch and we expect new models to use this framework as well. Currently, it is
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
       def __init__(self, vaes, model_config=None):
           """
           Multimodal Variaional Autoencoder with Product of Experts https://github.com/mhw32/multimodal-vae-public
           :param vaes: list of modality-specific vae objects
           :type encoders: list
           :param model_cofig: config with model-specific parameters
           :type model_config: dict
           """
           super().__init__()
           self.vaes = nn.ModuleDict(vaes)
           self.model_config = model_config
           self.modelName = 'poe'
           self.pz = dist.Normal
           self.prior_dist = dist.Normal


The TorchMMVAE class includes the bare functional minimum for a multimodal VAE, i.e. the forward pass and multimodal_mixing function.
The newly added model can override both these methods or keep the forward function as it is. Here we add the ``forward()`` pass and all methods necessary for the multimodal data integration. The first input parameter
will be the multimodal data specified in a config where the keys label the modalities and values contain the data (and possibly masks where applicable).

.. code-block:: python
   :emphasize-lines: 1

    def forward(self, inputs, K=1):
        """
        Forward pass that takes input data and outputs a dict iwth  posteriors, reconstructions and latent samples
        :param inputs: input data, a dict of modalities where missing modalities are replaced with None
        :type inputs: dict
        :param K: sample K samples from the posterior
        :type K: int
        :return: dict where keys are modalities and values are a named tuple
        :rtype: dict
        """
        mu, logvar, single_params = self.infer(inputs, K)
        qz_x = dist.Normal(*[mu, logvar])
        z = qz_x.rsample(torch.Size([1]))
        qz_d, px_d, z_d = {}, {}, {}
        z_d["joint"] = {"latents": z, "masks": None}
        for mod, vae in self.vaes.items():
            px_d[mod] = vae.px_z(*vae.dec(z_d["joint"]))
        output_dict = {}
        qz_d["joint"] = qz_x
        for modality in self.vaes.keys():
            output_dict[modality] = VaeOutput(encoder_dists=qz_d["joint"], decoder_dists=[px_d[modality]],
                                                latent_samples=z_d["joint"])
        return output_dict

    def infer(self,x, K=1):
        """
        Inference module, calculates the joint posterior
        :param inputs: input data, a dict of modalities where missing modalities are replaced with None
        :type inputs: dict
        :param K: sample K samples from the posterior
        :type K: int
        :return: joint posterior and individual posteriors
        :rtype: tuple(torch.tensor, torch.tensor, list, list)
        """
        for key in x.keys():
            if x[key]["data"] is not None:
                batch_size = x[key]["data"].shape[0]
                break
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.vaes["mod_1"].n_latents), use_cuda=True)
        for m, vae in self.vaes.items():
            if x[m]["data"] is not None:
                mod_mu, mod_logvar = vae.enc(x[m])
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)
        mu_before, logvar_before = mu, logvar
        # product of experts to combine gaussians
        mu, logvar = self.product_of_experts(mu, logvar)
        return mu, logvar, [mu_before[1:], logvar_before[1:]]

    def product_of_experts(self, mu, logvar):
        """
        Calculated the product of experts for input data
        :param mu: list of means
        :type mu: list
        :param logvar: list of logvars
        :type logvar: list
        :return: joint posterior
        :rtype: tuple(torch.tensor, torch.tensor)
        """
        eps = 1e-8
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = pd_var
        return pd_mu, pd_logvar

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


The ``forward()`` method should return a dictionary with keys denoting individual modalities ("mod_1", ..., "mod_n"). Each
key contains the named tuple VaeOutput which has the following keys
* encoder_dists - a list of posteriors (parametrized torch.dist objects)
* decoder_dists - a list of output likelihood distributions (parametrized torch.dist objects)
* latent_samples - sampled latent vectors ``z`` (will be used for log-likelihood calculation)

If you defined the forward function, it is the minimal scenario you need for a functional model.

Next, we need to add our model to the list of all models in ``__init__.py`` located in the ``models`` directory:

.. code-block:: python
   :emphasize-lines: 2, 6

    from .mmvae_models import MOE as moe
    from .mmvae_models import POE as poe
    from .mmvae_models import MoPOE as mopoe
    from .mmvae_models import DMVAE as dmvae
    from .vae import VAE
    __all__ = [dmvae, moe, poe, mopoe, VAE]

If we need to, we can define specific encoder and decoder networks (although we can also re-use the existing ones).

The last thing we need to do is to add the objective function into ``objectives.py``. The name should be in the form
"multimodal_<objectivename>_<modelname>" so that the toolkit knows when to use it. "objectivename" will be defined in the config
as the ``obj`` key, modelname will be the ``mixing`` method.


.. code-block:: python

   def multimodal_elbo_poe(model, x,  ltype="lprob"):
       """Subsampled ELBO with the POE approach as used in https://github.com/mhw32/multimodal-vae-public

       :param model: model object
       :type model: object
       :param x: input batch
       :type x: torch.tensor
       :param ltype: reconstruction loss term
       :type ltype: str
       :return: loss, kl divergence, reconstruction loss
       :rtype: tuple(torch.tensor, torch.tensor, list)
       """
       lpx_zs, klds, elbos = [[] for _ in range(len(x))], [], []
       for m in range(len(x.keys()) + 1):
           mods = copy.deepcopy(x)
           for d in mods.keys():
               mods[d]["data"] = None
               mods[d]["masks"] = None
           if m == len(x.keys()):
               mods = x
           else:
               mods["mod_{}".format(m+1)] = x["mod_{}".format(m+1)]
           output_dic = model(mods)
           kld = kl_divergence(output_dic["mod_1"].encoder_dists, model.pz(*model.vaes["mod_1"]._pz_params))
           klds.append(kld.sum(-1))
           loc_lpx_z = []
           for mod in output_dic.keys():
               px_z = output_dic[mod].decoder_dists[0]
               lpx_z = (loss_fn(px_z, x[mod]["data"], ltype=ltype, categorical=x[mod]["categorical"]) * model.vaes[d].llik_scaling).sum(-1)
               loc_lpx_z.append(lpx_z)
               if mod == "mod_{}".format(m+1):
                   lpx_zs[m].append(lpx_z)
           elbo = (torch.stack(loc_lpx_z).sum(0) - kld.sum(-1).sum())
           elbos.append(elbo)
       individual_losses = [-torch.stack(m).sum() / model.vaes["mod_{}".format(idx+1)].llik_scaling for idx, m in enumerate(lpx_zs)]
       return -torch.stack(elbos).sum(), torch.stack(klds).mean(0).sum(), individual_losses

The objective function receives the model object on the inpt as well as the training/testing data, number of K samples
if used and type of the reconstruction loss term (we use it as a param to the ``loss_fn`` which calculates the reconstrction
loss). We want the objective to return the final loss (``torch.float32``) and optionally KLD loss (``torch.float32``) and
modality-specific losses (a list with ``torch.float32``).

Now we should be able to train using this model. We need to create a ``config.yml`` file as follows:

.. code-block:: yaml
   :emphasize-lines: 7, 16,17,22,23

   batch_size: 32
   epochs: 700
   exp_name: poe_exp
   labels: ./data/mnist_svhn/labels.pkl
   loss: lprob
   lr: 1e-3
   mixing: poe
   n_latents: 10
   obj: elbo
   optimizer: adam
   pre_trained: null
   seed: 2
   viz_freq: 20
   test_split: 0.1
   modality_1:
      decoder: MNIST
      encoder: MNIST
      feature_dim: [28,28,1]
      mod_type: image
      path: ./data/mnist_svhn/mnist
   modality_2:
      decoder: SVHN
      encoder: SVHN
      feature_dim: [32,32,3]
      mod_type: image
      path: ./data/mnist_svhn/svhn

You can see that we specified "poe" as our multimodal mixing model. After configuring the experiment, we can run the training:


.. code-block:: python

   cd multimodal-compare
   python main.py --cfg ./configs/config.yml

