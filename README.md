# Multimodal VAE module
This repository contains an adapted version of the **Variational Mixture-of-Experts Autoencodersfor Multi-Modal Deep Generative Models** (see [paper](https://arxiv.org/pdf/1911.03393.pdf) or [github](https://github.com/iffsid/mmvae)), with additional option to train using **Multimodal generative models for scalable weakly-supervised learning** (see [paper](https://arxiv.org/pdf/1802.05335.pdf) or [github](https://github.com/mhw32/multimodal-vae-public)). The purpose of this repo is to fuse data of various types into one joint latent representation, so that we can generate samples or reconstruct one modality from another. The repository will be soon remade into a ROS2 package.

## Requirements
The list of packages needed can be found in `requirements.txt`. However, we recommend to install a conda env from environment.yml:


`conda env create -f environment.yml`

`conda activate mmvae`



## Usage

### Generate dummy dataset

The code currently supports either images of size [64,64,3] or .pkl with arrays of format [dataset_size, 1D_array], the arrays can have an arbitrary size. 
We provide code to generate such dummy dataset, i.e. a set of images with colored objects, another matching set of images with corresponding color names and an attrs.pkl file with single number arrays
that represent the color label. You can train with any pair from these three "modalities", just add the path to the dataset in appropriate arguments in the config. To generate your custom dataset, run:


`python generate_dataset.py --size 10000`


You can make the object colors or color names noisy by adding --noisytxt or --noisycol arguments. The generated data will be saved in the /data folder and is ready to train.


**!!** Alternatively, you can generate a dataset consisting od images and corresponding word embeddings. To learn how to do that and see the dataset examples, see the [**Wiki page**](https://gitlab.ciirc.cvut.cz/imitrob/mirracle/mirracle_wiki/-/wikis/tutorials/How-to-train-and-run-Multimodal-Fusion)


### Training

The parameters can be set up via a .yml config or command line arguments - these are the same ones, but override the config. To train using the config, run:


```bash
python main.py --cfg config1.yml

```


The content of the config is following:
```
general:
  n_epochs: 500  # number of epochs to train
  n_latents: 8   # size of the n-dimensional latent vector
  obj: moe_elbo  # objective, for single modality training, use elbo/iwae/dreg. For multiple modalities, use moe_elbo/poe_elbo/poe_elbo_semi (produces better results)/iwae/dreg
  loss: lprob    # how to calculate the loss, "lprob" (log_prob of a distribution) is better when using the MoE approach, for PoE better use "bce" (binary cross entropy)
  viz_freq: 100  # save visualizations every n epochs
  model: 2mods   # if you train a bi-modal vae, set "2mods", for unimodal vae, set "uni" - this will take the dataset from modality_1 only
  noisy_txt: False  # if you have numeric/text labels as one modality, you can make them noisy (for experimental reasons only)
modality_1:
  dataset: ../data/image  # path to the folder or .pkl file with the first modality train data
  type: img # how to treat the modality, mostly for logging/saving reasons, does not depend on the data type
modality_2:
  dataset: ../data/attrs.pkl  # path to the folder or .pkl file with the second modality train data
  type: txt # how to treat the modality, mostly for logging/saving reasons, does not depend on the data type
```

The arguments contain some additional hyperparameters which were included in the MoE code, however we did not yet test these so they remain default:

- **`--K`**: Number of particles, controls the number of particles `K` in IWAE/DReG estimator (see the paper).
- **`--learn-prior`**: Prior variance learning, controls whether to enable prior variance learning. Results in our paper are produced with this enabled. Excluding this argument in the command will disable this option;
- **`--llik_scaling`**: Likelihood scaling, specifies the likelihood scaling of one of the two modalities, so that the likelihoods of two modalities contribute similarly to the lower bound. The default values are: 
    - _MNIST-SVHN_: MNIST scaling factor 32*32*3/28*28*1 = 3.92
    - _CUB Image-Cpation_: Image scaling factor 32/64*64*3 = 0.0026


### Analysing
After training, various reconstructions, sampling and embedding visualizatoins will be saved in the results directory. You can additionally analyse the data using these scripts:

- for likelihood estimation of data using a trained model, run `python calculate_likelihoods.py --save-dir path/to/trained/model/folder/ --iwae-samples 1000`;
- for coherence analysis and latent digit classification accuracy on MNIST-SVHN dataset, run `python analyse_ms.py --save-dir path/to/trained/model/folder/`;
- for coherence analysis on CUB image-caption dataset, run `python analyse_cub.py --save-dir path/to/trained/model/folder/`.
