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
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/level2.zip   # replace level2 with any of the 1-5 levels
unzip level2.zip -d ./data/
```

![Examples of levels 1, 3 and 5](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/dataset.png "GeBiD dataset")

### Dataset generation

Alternatively, you can generate a dataset on your own. For the default configuration, run for example:

 ```
cd ~/multimodal-vae-comparison/multimodal_compare
python ./data_proc/generate_dataset.py --dir ./data/level4 --level 4 --size 10000 
```

The code will make an _./image_ folder in the target directory that includes the _.png_ images. The text is stored in 
_attrs.pkl_ file and is in the same order as the images. 

## Setup and training

### Single experiment
We show an example training config in _./multimodal_compare/configs/config1.yml_. You can run the training as follows:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python main.py --cfg configs/config1.yml
```

### Set of experiments

We provide an automated way to perform hyperparameter grid search for your models. First, set up the default config (e.g. _config1.yml_ in _./configs_)
that should be adjusted in selected parameters. Then generate the full variability within the chosen parametes as follows:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python data_proc/generate_configs.py --path ./configs/my_experiment  --cfg ./configs/config1.yml --n-latents 24 32 64 --mixing moe poe --seed 1 2 3 
```

The script will make 18 configs (2 models x 3 seeds x 3 latent dimensionalities) within the chosen directory. To see the full 
spectrum of parameters that can be adjusted, run:

```python data_proc/generate_configs.py -h```

To automatically run the whole set of experiments located in one folder, launch:

```./iterate_configs.sh "./configs/my_experiment/*" ```

We provide sets of configs for the experiments reported in the paper. These are located in _./configs/batch_size_exp_
and  _./configslatent_dim_exp_


## Evaluation

After training, you will find various visualizations of the training progress in the _./visuals_ folder of your experiment.
Furthermore, to calculate the joint- and cross-generation accuracy, you can run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python eval/eval_gebid.py --m /path/to/model/directory --level 4  # specify the level on which the model was trained
```

## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  


## Acknowledgment

The toolkit features models and functions from the official implementations of [MVAE](https://github.com/mhw32/multimodal-vae-public) ([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) ([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) ([paper](https://openreview.net/forum?id=5Y21V0RDBV)).

## Contact

For any additional questions, feel free to email [sejnogab@fel.cvut.cz](mailto:sejnogab@fel.cvut.cz) 
