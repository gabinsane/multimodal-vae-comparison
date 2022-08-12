.. _addmodel:

Add a new model
====================

We encourage the authors to implement their own multimodal VAE models into our toolkit. Here we describe how to do it.

General requirements
---------------------

The toolkit is written using PyTorch and we expect new models to use this framework as well. Currently, it is
possible to implement unimodal VAEs and any multimodal VAEs which use dedicated VAE instances for each modality.
You can add a new objective, encoder/decoder networks and of course other support modules that are needed.

Below we show a step-by-step tutorial in which we add the DMVAE model. You can find the original implementation
in their `GitHub repository <https://github.com/seqam-lab/DMVAE>`_.


Adding a new model
---------------------

First, we start by defining the model in ``mmvae_models.py``. Our model will need a name and should inherit the MMVAE
class defined in ``mmvae_base.py``. This class handles some common functions for dataset loading, sample visualisation etc.
The init parameters are passed from the config file and should have this fixed structure.
``self.modelName`` will be used for model selection from the config file.

.. code-block:: python
   :emphasize-lines: 1, 3

    class DMVAE(MMVAE):
        """Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations https://github.com/seqam-lab/DMVAE"""
        def __init__(self, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size):
            self.modelName = 'dmvae'
            super(DMVAE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size)
            self.n_latents = n_latents
            self.qz_x = dist.Normal

Now we need to add the ``forward()`` pass and all methods necessary for the multimodal data integration. The first input parameter
will be the multimodal data concatenated in a list, possible missing modalities are replaced with ``None``. You can add
other input parameters in case you only want to use this model with your custom objective which you will define later.

.. code-block:: python
   :emphasize-lines: 1

    def forward(self, x):
        qz_xs_shared, px_zs= [], [[None for _ in range(len(self.vaes)+1)] for _ in range(len(self.vaes))]
        qz_xs_private = [None for _ in range(len(self.vaes))]
        # initialise cross-modal matrix
        for m, vae in enumerate(self.vaes):
            if x[m] is not None:
                mod_mu, mod_std = self.vaes[m].enc(x[m].to("cuda") if not isinstance(x[m], list) else x[m])
                qz_xs_private[m] = self.vaes[m].qz_x(*[mod_mu[0], mod_std[0]])
                qz_xs_shared.append(self.vaes[m].qz_x(*[mod_mu[1], mod_std[1]]))
        mu_joint, std_joint = self.apply_poe(qz_xs_shared)
        joint_d = self.qz_x(*[mu_joint, std_joint])
        all_shared = qz_xs_shared + [joint_d]
        zss = []
        for d, vae in enumerate(self.vaes):
            for e, dist in enumerate(all_shared):
                    zs_shared = dist.rsample(torch.Size([K]))
                    zs_private = qz_xs_private[d].rsample(torch.Size([K]))
                    zs = torch.cat([zs_private, zs_shared], -1)[0]
                    zss.append(zs)
                    if "transformer" in vae.dec_name.lower():
                        px_zs[d][e] = vae.px_z(*vae.dec([zs, x[d][1]] if x[d] is not None else [zs, None]))
                    else:
                        px_zs[d][e] = vae.px_z(*vae.dec(zs))
        return qz_xs_private + all_shared, px_zs, zss

The ``forward()`` method should return a tuple of 3 lists:
* a list of posteriors (parametrized torch.dist objects)
* a list of output likelihood distributions (parametrized torch.dist objects)
* a list of sampled latent vectors ``z`` (will be used for log-likelihood calculation)

If you defined the forward function, it is the minimal scenario you need for a functional model.

Next, we need to add our model to the list of all models in ``__init__.py`` located in the ``models`` directory:

.. code-block:: python
   :emphasize-lines: 4, 6

    from .mmvae_models import MOE as moe
    from .mmvae_models import POE as poe
    from .mmvae_models import MoPOE as mopoe
    from .mmvae_models import DMVAE as dmvae
    from .vae import VAE
    __all__ = [dmvae, moe, poe, mopoe, VAE]

If we need to, we can define specific encoder and decoder networks (although we can also re-use the existing ones).
Here we define a new encoder network for the MNIST dataset. This can be added into ``encoders.py``

.. code-block:: python

    class Enc_MNIST_DMVAE(nn.Module):
        def __init__(self, latent_dim, data_dim, num_pixels=784, num_hidden=256, zPrivate_dim=1):
            super(Enc_MNIST_DMVAE, self).__init__()
            self.net_type = "FNN"
            temp = 0.66
            self.digit_temp = torch.tensor(temp)
            self.zPrivate_dim = zPrivate_dim
            self.zShared_dim = latent_dim
            self.enc_hidden = nn.Sequential(
                nn.Linear(num_pixels, num_hidden),
                nn.ReLU())
            self.fc = nn.Linear(num_hidden, 2 * zPrivate_dim + 2 * latent_dim)

        def forward(self, x):
            hiddens = self.enc_hidden(x.reshape(1,x.shape[0], -1).float())
            stats = self.fc(hiddens)
            muPrivate = stats[:, :, :self.zPrivate_dim]
            logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
            stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + Constants.eps)
            muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
            logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
            stdShared = torch.sqrt(torch.exp(logvarShared) + Constants.eps)
            return [muPrivate, muShared], [stdPrivate, stdShared]

Again, we need to intialize the class with training parameters like ``latent_dim`` (size of the latent space) and
``data_dim`` (the expected dimensionality of the data defined in config).

The name of the encoder will be the class name after the underscore, i.e. "MNIST_DMVAE" in this case. You can use this
in the config to select this network.

Normally, the ``forward()`` method would return a tuple of means and standard deviations (torch.tensors). However, in this case
we have two types of means and stds (private and shared) and thus adjust the output as needed.

Now we need to also add a decoder in ``decoders.py``:

.. code-block:: python
   :emphasize-lines: 4, 6

    class Dec_MNIST_DMVAE(nn.Module):
        def __init__(self, latent_dim, data_dim, num_pixels=784, num_hidden=256, zPrivate_dim=1):
            super(Dec_MNIST_DMVAE, self).__init__()
            self.net_type = "FNN"
            self.style_mean = zPrivate_dim
            self.style_std = zPrivate_dim
            self.num_digits = latent_dim

            self.dec_hidden = nn.Sequential(
                nn.Linear(zPrivate_dim + latent_dim, num_hidden),
                nn.ReLU())
            self.dec_image = nn.Sequential(
                nn.Linear(num_hidden, num_pixels),
                nn.Sigmoid())

        def forward(self, z):
            hiddens = self.dec_hidden(z)
            x = self.dec_image(hiddens)
            return x, torch.tensor(0.75).to(x.device)

The process is similar to the encoder. The output here are the modality reconstructions. Here we could also add other
encoders and decoder for different modalities, perhaps SVHN in this scenario.

The last thing we need to do is to add the objective function into ``objectives.py``. The name should be in the form
"multimodal_<objectivename>_<modelname>" so that the toolkit knows when to use it. "objectivename" will be defined in the config
as the ``obj`` key, modelname will be the ``mixing`` method.


.. code-block:: python

    def multimodal_elbo_dmvae(model, x, K=1, ltype="lprob"):
        "Objective for the DMVAE model. Source: https://github.com/seqam-lab/DMVAE"
        qz_xs, px_zs, zss = model(x)
        recons = []
        kls = []
        ind_losses = []
        for i in range(len(px_zs)):
            for j in range(len(px_zs[i])):
                if j < len(px_zs[i])-1:
                    recons.append(loss_fn(px_zs[i][j], x[i], ltype=ltype, mod_type=model.vaes[i].dec_name).cuda() * model.vaes[i].llik_scaling.mean())
                else:
                    recons.append(torch.tensor(0))
            for n in range(len(px_zs[i])-1):
                idxs = [2+n,4]
                log_pz = log_joint([px_zs[i][n], px_zs[n+1]], [zss[i], zss[idxs[n]]])
                log_q_zCx = log_joint([qz_xs[i], qz_xs[idxs[n]]], [zss[i], zss[idxs[n]]])
                log_qz, _, log_prod_qzi = log_batch_marginal([qz_xs[i], qz_xs[idxs[n]]])
                kl = ((log_q_zCx - log_qz) *(log_qz - log_prod_qzi)* (log_prod_qzi - log_pz)).mean()
                kls.append(kl)
        # cross sampling
        for i in get_all_pairs(px_zs):
            recons.append(loss_fn(px_zs[i[0]][0], x[i[0]], ltype=ltype, mod_type=model.vaes[i].dec_name).cuda() * model.vaes[
                i].llik_scaling.mean())
            log_pz = log_joint([px_zs[i[0]][0], px_zs[i[0]][1]])
            log_q_zCx = log_joint([qz_xs[i[0]][0], qz_xs[i[0]][1]])
            log_qz, _, log_prod_qzi = log_batch_marginal([qz_xs[i[0]][0], qz_xs[i[0]][1]])
            kl =  ((log_q_zCx - log_qz) *(log_qz - log_prod_qzi)* (log_prod_qzi - log_pz)).mean()
            kls.append(kl)
        for rec, kl in zip(recons, kls):
            l = rec - kl
            ind_losses.append(l)
        loss = torch.tensor(ind_losses).sum()
        return -loss, torch.stack(kls).sum(), ind_losses

The objective function receives the model object on the inpt as well as the training/testing data, number of K samples
if used and type of the reconstruction loss term (we use it as a param to the ``loss_fn`` which calculates the reconstrction
loss). We want the objective to return the final loss (``torch.float32``) and optionally KLD loss (``torch.float32``) and
modality-specific losses (a list with ``torch.float32``).

Now we should be able to train using this model. We need to create a ``config.yml`` file as follows:

.. code-block:: yaml
   :emphasize-lines: 7, 16,17,22,23

   batch_size: 32
   epochs: 700
   exp_name: dmvae_exp
   labels: ./data/mnist_svhn/labels.pkl
   loss: lprob
   lr: 1e-3
   mixing: dmvae
   n_latents: 10
   obj: elbo
   optimizer: adam
   pre_trained: null
   seed: 2
   viz_freq: 20
   test_split: 0.1
   modality_1:
      decoder: MNIST_DMVAE
      encoder: MNIST_DMVAE
      feature_dim: [28,28,1]
      mod_type: image
      path: ./data/mnist_svhn/mnist
   modality_2:
      decoder: SVHN_DMVAE
      encoder: SVHN_DMVAE
      feature_dim: [32,32,3]
      mod_type: image
      path: ./data/mnist_svhn/svhn

You can see that we specified "dmvae" as our multimodal mixing model and selected the previously defined encoder and
decoder networks. After configuring the experiment, we can run the training:


``
cd multimodal-compare

python main.py --cfg ./configs/config.yml
``