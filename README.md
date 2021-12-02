# Multimodal VAE module
This repository contains an adapted version of the **Variational Mixture-of-Experts Autoencodersfor Multi-Modal Deep Generative Models** (see [paper](https://arxiv.org/pdf/1911.03393.pdf) or [github](https://github.com/iffsid/mmvae)), with additional option to train using **Multimodal generative models for scalable weakly-supervised learning** (see [paper](https://arxiv.org/pdf/1802.05335.pdf) or [github](https://github.com/mhw32/multimodal-vae-public)). The purpose of this repo is to fuse data of various types into one joint latent representation, so that we can generate samples or reconstruct one modality from another. The repository will be soon remade into a ROS2 package.

## Requirements
The list of packages needed can be found in `requirements.txt`. However, we recommend to install a conda env from environment.yml:


`conda env create -f environment.yml`

`conda activate mmvae`



## Usage

## Generate dummy dataset

Currently you can generate two types of datasets, _img-img_ (10 classes) and _img-vec_ (10x5x2 classes)


**Img-img** comprises two sets of images [64,64,3], one set contains a rectangle of various colors (10 altogether), the other one contains the name of the corresponding color imprinted into the image [64,64,3]. 


**Img-vec** comprises a set of images [64,64,3] and a set of corresponding word embeddings [3, 4096]. The image set in this case contains 5 different shapes, 10 different colors and 2 different sizes. The descriptions are in the form of [size, color, shape].  


Since there are 3 words for each image and each has a 4096D embedding, the overall shape is 12288 - that is the same as the overall dimensionality of the images [64x64x3]. The embeddings are generated using a custom pre-trained word2vec model. To see how it was done, see `train_w2v.py` in _data_proc_ directory.


For each dataset, there is also an attrs.pkl file with text label annotations (color name for img-img and the three word description for img-vec).

To generate the datasets, run:

```
cd mirracle_multimodal/mirracle_multimodal
python data_proc/generate_dataset.py --size 10000 --type img-vec
```

The dataset will be generated in the *mirracle_multimodal/data* folder.  


### Training

For learning how to train on CIIRC cluster, see [**Wiki page**](https://gitlab.ciirc.cvut.cz/imitrob/mirracle/mirracle_wiki/-/wikis/tutorials/How-to-train-and-run-Multimodal-Fusion)


The parameters can be set up via a .yml config or command line arguments - these are the same ones, but override the config. To train using the config, run:

To train with single modality (image example here), run:
```bash
python main.py --cfg config1mod.yml
```

To train bimodal VAE (image and text here), run:
```bash
python main.py --cfg config2mods.yml
```
You can change the mixing parameter here from "poe" to "moe" to use Mixture-of-Experts instead of Product-of-Experts


The content of the config is following:
```
general:
[general]
  n_latents: 24  # size of the n-dimensional latent vector
  batch_size: 64
  epochs: 500  # number of epochs to train
  obj: elbo # objective, only "elbo" is fully supported now
  loss: bce # reconstruction loss, either lprob or bce. bce is more stable
  viz_freq: 100  # save visualizations every n epochs
  mixing: poe # multimodal mixing approach. either "moe" or "poe"
  modalities_num: 2 # how many modalities to combine from the ones you fill below
  exp_name: test # folder name to save the model in
  pre_trained:  # you can add a path to a pretrained model to load as starting weights
  llik_scaling: 0 # serves for balancing modalities with larger dimensionality. 0 is automatic balancing, 1 is no balancing
  seed: 1 # fixed random seed
  mod1_type: img # how to treat the modality, at the moment use img or txt
  mod1_path: ../data/image # path to the first modality dataset (folder or .pkl file)
  mod2_type: txt
  mod2_path:../data/3d.pkl
  mod2_numwords: 3 # if we use text encodings, enter the number of words in each sequence
  mod3_type:
  mod4_type:
  mod5_type:
  mod6_type:
```

### Analysing
After training, various reconstructions, sampling and embedding visualizatoins will be saved in the results directory. You can perform further analysis using the script `infer.py`
