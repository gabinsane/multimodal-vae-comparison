from ..mmvae_base import TorchMMVAE
import torch.distributions as dist


class ExampleTorchVAE(TorchMMVAE):
    def __init__(self):
        super(ExampleTorchVAE).__init__()

    def forward(self, inputs, K=1):
        pass


class MOE(TorchMMVAE):
    modelname = 'moe'

    def __init__(self, encoders, decoders, data_paths, feature_dims, mod_types, n_latents, test_split, batch_size):
        """
        Multimodal Variaional Autoencoder with Mixture of Experts https://github.com/iffsid/mmvae

        :param encoders: list of encoder names (strings) as listed in config
        :type encoders: list
        :param decoders: list of decoder names (strings) as listed in config
        :type decoders: list
        :param data_paths: list of data paths for all modalities
        :type data_paths: list
        :param feature_dims: list of modality-specific feature dimensions as listed in config
        :type feature_dims: list
        :param mod_types: list of modality types (strings) from config
        :type mod_types: list
        :param n_latents: list of latent dimensionalities from config
        :type n_latents: list
        :param test_split: fraction of the data to be used for validation
        :type test_split: float
        :param batch_size: batch size
        :type batch_size: int
        """
        self.modelName = 'moe'
        super(MOE, self).__init__(dist.Normal, encoders, decoders, data_paths, feature_dims, mod_types, n_latents,
                                  test_split, batch_size)

    def get_missing_modalities(self, mods):
        """
        Get indices of modalities that are missing

        :param mods: list of modalities
        :type mods: list
        :return: list of indices of missing modalities
        :rtype: list
        """
        indices = []
        for i, e in enumerate(mods):
            if e is None:
                indices.append(i)
        return indices

    def forward(self, x, K=1):
        """
        Forward pass that takes input data and outputs a list of posteriors, reconstructions and latent samples

        :param x: input data, a list of modalities where missing modalities are replaced with None
        :type x: list
        :param K: sample K samples from the posterior
        :type K: int
        :return: a list of posterior distributions, a list of reconstructions and latent samples
        :rtype: tuple(list, list, list)
        """
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            if x[m] is not None:
                qz_x, px_z, zs = vae(x[m], K=K)
                qz_xs.append(qz_x)
                zss.append(zs)
                px_zs[m][m] = px_z  # fill-in diagonal
        for ind in self.get_missing_modalities(x):
            lat = zss[0] if "transformer" not in self.vaes[ind].dec_name.lower() else [zss[0], None]
            px_zs[ind][ind] = self.vaes[ind].px_z(*self.vaes[ind].dec(lat))
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if "transformer" in self.vaes[d].dec_name.lower():
                        px_zs[e][d] = vae.px_z(*vae.dec([zs, x[d][1]] if x[d] is not None else [zs, None]))
                    else:
                        px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def reconstruct(self, data, runPath, epoch, N=8):
        """
        Reconstruct data for individual experts

        :param data: list of input modalities
        :type data: list
        :param runPath: path to save data to
        :type runPath: str
        :param epoch: current epoch to name the data
        :type epoch: str
        :param N: how many samples to reconstruct
        :type N: int
        """
        recons_mat = super(MOE, self).reconstruct([d for d in data])
        self.process_reconstructions(recons_mat, data, epoch, runPath)
