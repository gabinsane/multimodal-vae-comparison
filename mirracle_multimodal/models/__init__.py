from .mmvae_cub_images_sentences import CUB_Image_Sentence as VAE_cubIS
from .mmvae_cub_images_sentences_ft import CUB_Image_Sentence_ft as VAE_cubISft
from .mmvae_own import MOE as VAE_2_moe
from .mmvae_own import POE as VAE_2_poe
from .vae_cub_image import CUB_Image as VAE_cubI
from .vae_cub_image_ft import CUB_Image_ft as VAE_cubIft
from .vae_cub_sent import CUB_Sentence as VAE_cubS
from .vae_mnist import MNIST as VAE_mnist
from .vae_mnist import CROW as VAE_imgcol
from .vae_svhn import SVHN as VAE_svhn
from .vae_svhn import CROW2 as VAE_imgtxt
from .vae_own import UNIVAE as VAE_1

__all__ = [VAE_2_moe, VAE_mnist, VAE_svhn, VAE_cubIS, VAE_cubS,
           VAE_cubI, VAE_cubISft, VAE_cubIft, VAE_imgcol, VAE_imgtxt, VAE_2_poe,
           VAE_1]
