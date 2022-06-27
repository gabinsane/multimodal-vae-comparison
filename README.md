# Multimodal VAE Comparison

This is the official code for the submitted NIPS 2022 Datasets and Benchmarks paper "Benchmarking Multimodal Variational Autoencoders: GeBiD Dataset and Toolkit".

The purpose of this toolkit is to offer a systematic and unified way to train, evaluate and compare the state-of-the-art
multimodal variational autoencoders. The toolkit can be used with arbitrary datasets and both uni/multimodal settings.
By default, we provide implementations of the [MVAE](https://github.com/mhw32/multimodal-vae-public) 
([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) 
([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) 
([paper](https://openreview.net/forum?id=5Y21V0RDBV)) models, but anyone is free to contribute with their own
implementation. 

We also provide a custom synthetic bimodal dataset, called GeBiD, designed specifically for comparison of the
joint- and cross-generative capabilities of multimodal VAEs. You can read about the utilities of the dataset in the proposed 
paper (link will be added soon). This dataset offers 5 levels of difficulty (based on the number of attributes)
to find the minimal functioning scenario for each model. Moreover, its rigid structure enables automatic qualitative
evaluation of the generated samples. For more info, see below. 

## Preliminaries

This code was tested with:

- Python version 3.6.8
- PyTorch version 1.10.1
- CUDA version 10.2

We recommend to install the conda enviroment as follows:

```
conda env create -f environment.yml
conda activate multivae                 
```

## Get the GeBiD dataset

We provide a bimodal image-text dataset GeBiD (Geometric shapes Bimodal Dataset) for systematic multimodal VAE comparison. There are 5 difficulty levels 
based on the number of featured attributes (shape, size, color, position and background color). You can either generate
the dataset on your own, or download a ready-to-go version.

### Dataset download 
You can download any of the following difficulty levels: [Level 1](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level1.zip),
[Level 2](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level2.zip), [Level 3](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level3.zip),
[Level 4](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level4.zip), [Level 5](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level5.zip).

The dataset should be placed in the ./data directory. For downloading, unzipping and moving the chosen dataset, run:

```
cd ~/mirracle_multimodal/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level2.zip   # replace level2 with any of the 1-5 levels
unzip level2.zip -d ./data/
```

![Examples of GeBiD levels](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/gebid_dataset.png "GeBiD dataset")

### Dataset generation

Alternatively, you can generate a dataset on your own. For the default configuration, run for example:

 ```
cd ~/mirracle_multimodal/multimodal_compare
python ./data_proc/generate_dataset.py --dir ./data/level4 --level 4 --size 10000 
```

The code will make an _./image_ folder in the target directory that includes the _.png_ images. The text is stored in 
_attrs.pkl_ file and is in the same order as the images. 

## Setup and training

### Single experiment
We show an example training config in _./multimodal_compare/configs/config1.yml_. You can run the training as follows:

```
cd ~/mirracle_multimodal/multimodal_compare
python main.py --cfg configs/config1.yml
```

The config contains general arguments and modality-specific arguments (denoted as "modality_n"). In general, you can set up a training for 1-N modalities by defining the required subsections for each of them. 
The paths to all modalities are expected to have the data ordered so that they are semantically matching (e.g. the first image and the first text sample belong together).

The usage and possible options for all the config arguments are below:

![Config documentation](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/config_ex.png "config documentation")

### Set of experiments

We provide an automated way to perform a hyperparameter grid search for your models. First, set up the default config (e.g. _config1.yml_ in _./configs_)
that should be adjusted in the selected parameters. Then generate the full variability within the chosen parameters as follows:

```
cd ~/mirracle_multimodal/multimodal_compare
python data_proc/generate_configs.py --path ./configs/my_experiment  --cfg ./configs/config1.yml --n-latents 24 32 64 --mixing moe poe --seed 1 2 3 
```

The script will make 18 configs (2 models x 3 seeds x 3 latent dimensionalities) within the chosen directory. To see the full 
spectrum of parameters that can be adjusted, run:

```python data_proc/generate_configs.py -h```

To automatically run the whole set of experiments located in one folder, launch:

```./iterate_configs.sh "./configs/my_experiment/*" ```

We provide sets of configs for the experiments reported in the paper. These are located in _./configs/batch_size_exp_
and  _./configs/latent_dim_exp_


### Training on other datasets

By default, we also support training on MNIST_SVHN (or MNIST/SVHN only) and the Caltech-UCSD Birds 200 (CUB) dataset as 
used in the [MMVAE paper](https://arxiv.org/pdf/1911.03393.pdf). We provide the default training configs which
 you can adjust according to your needs (e.g. change the model, loss objective etc.). 


For training on MNIST_SVHN, first download the dataset (30 MB in total) before the training. You can run the following:

```
cd ~/mirracle_multimodal/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/mnist_svhn.zip   # download mnist_svhn dataset
unzip mnist_svhn.zip -d ./data/
python main.py --cfg configs/config_mnistsvhn.yml
```

For training on CUB, we provide our preprocessed and cleaned version of the dataset (106 MB in total). To download and train, run:

```
cd ~/mirracle_multimodal/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/cub.zip   # download CUB dataset
unzip cub.zip -d ./data/
python main.py --cfg configs/config_cub.yml
```
## Evaluation

After training, you will find various visualizations of the training progress in the _./visuals_ folder of your experiment.
Furthermore, to calculate the joint- and cross-generation accuracy, you can run:

```
cd ~/mirracle_multimodal/multimodal_compare
python eval/eval_gebid.py --m /path/to/model/directory --level 4  # specify the level on which the model was trained
```


## Extending for own models and networks

The toolkit is designed so that it enables easy extension for new models, objectives or encoder/decoder networks. 
<div style="text-align: left">
 <img align="right" src="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/uml.png" width="300" />
</div>

Here you can see the UML diagram of the _./models_ folder. The VAE class is initialized at every training (for both the uni/multimodal scenario) along with its
specified encoder and decoder network.
In the multimodal scenario, a new VAE class is instantiated for each modality by the MMVAE class (in mmvae_base.py). MMVAE is then wrapped by the selected
model in multimodal_models (MOE, POE, MoPOE classes), where the multimodal fusion is specified.

New encoder and decoder networks can be added in the corresponding scripts (encoders.py, decoders.py). For choosing these networks in the config,
use only the part of the class name following after the underscore (e.g. CNN for the class Enc_**CNN**).

The objectives and reconstruction loss terms are defined in objectives.py. By default, when you use more than one modality,
the model will use the objective named "multimodal_" together with the objective and model defined in the config (e.g., for elbo objective and poe model, the objective _multimodal_elbo_poe_ will be used.).
If you wish to add new objectives, keep the naming consistent with these rules so that it can be easily configured. 


## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  


## Acknowledgment

The toolkit features models and functions from the official implementations of [MVAE](https://github.com/mhw32/multimodal-vae-public) ([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) ([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) ([paper](https://openreview.net/forum?id=5Y21V0RDBV)).

## Contact

For any additional questions, feel free to email [sejnogab@fel.cvut.cz](mailto:sejnogab@fel.cvut.cz) 
