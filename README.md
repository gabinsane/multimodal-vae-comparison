# Multimodal VAE Comparison

This is the official code for the submitted NIPS 2022 Datasets and Benchmarks paper "Benchmarking Multimodal Variational Autoencoders: GeBiD Dataset and Toolkit".

The purpose of this toolkit is to offer a systematic and unified way to train, evaluate and compare the state-of-the-art
multimodal variational autoencoders. The toolkit can be used with arbitrary datasets and both uni/multimodal settings.
By default, we provide implementations of the [MVAE](https://github.com/mhw32/multimodal-vae-public) 
([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) 
([paper](https://arxiv.org/pdf/1911.03393.pdf)), [MoPoE](https://github.com/thomassutter/MoPoE) 
([paper](https://openreview.net/forum?id=5Y21V0RDBV)) and [DMVAE](https://github.com/seqam-lab/DMVAE) ([paper](https://github.com/seqam-lab/DMVAE)) models, but anyone is free to contribute with their own
implementation. 

We also provide a custom synthetic bimodal dataset, called **GeBiD**, designed specifically for comparison of the
joint- and cross-generative capabilities of multimodal VAEs. You can read about the utilities of the dataset in the proposed 
paper (link will be added soon). This dataset offers 5 levels of difficulty (based on the number of attributes)
to find the minimal functioning scenario for each model. Moreover, its rigid structure enables automatic qualitative
evaluation of the generated samples. For more info, see below. 

[**Code Documentation & Tutorials**](https://gabinsane.github.io/multimodal-vae-comparison)

---
### **List of contents**

* [Preliminaries](#preliminaries) <br>
* [GeBiD dataset](#get-the-gebid-dataset) <br>
* [Setup & Training](#setup-and-training) <br>
* [Evaluation](#evaluation)<br>
* [GeBiD leaderboard](#gebid-leaderboard)<br>
* [Training on other datasets](#training-on-other-datasets) <br>
* [Add own model](#extending-for-own-models-and-networks)<br>
* [Common installation problems](#common-installation-problems)<br>
* [License & Acknowledgement](#license)<br>
* [Contact](#contact)<br>
---
## Preliminaries

This code was tested with:

- Python version 3.8.13
- PyTorch version 1.12.1
- CUDA version 10.2 and 11.6

We recommend to install the conda enviroment as follows:

```
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
conda activate multivae                 
```

Please note that the framework depends on the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework which manages the model training and evaluation. 


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

![Examples of GeBiD levels](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/gebid_dataset.png "GeBiD dataset")

### Dataset generation

Alternatively, you can generate a dataset on your own. For the default configuration, run for example:

 ```
cd ~/multimodal-vae-comparison/multimodal_compare
python ./data_proc/generate_dataset.py --dir ./data/level2 --level 2 --size 10000 
```

The code will make an _./image_ folder in the target directory that includes the _.png_ images. The text is stored in 
_attrs.pkl_ file and is in the same order as the images. 

## Setup and training

### Single experiment
We show an example training config in _./multimodal_compare/configs/config1.yml_. You can run the training as follows (assuming you downloaded or generated the dataset level 2 above):

```
cd ~/multimodal-vae-comparison/multimodal_compare
python main.py --cfg ./configs/config1.yml
```

The config contains general arguments and modality-specific arguments (denoted as "modality_n"). In general, you can set up a training for 1-N modalities by defining the required subsections for each of them. 
The paths to all modalities are expected to have the data ordered so that they are semantically matching (e.g. the first image and the first text sample belong together).

The usage and possible options for all the config arguments are below:

![Config documentation](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/config.png "config documentation")

### Set of experiments

We provide an automated way to perform a hyperparameter grid search for your models. First, set up the default config (e.g. _config1.yml_ in _./configs_)
that should be adjusted in the selected parameters. Then generate the full variability within the chosen parameters as follows:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python data_proc/generate_configs.py --path ./configs/my_experiment  --cfg ./configs/config1.yml --n-latents 24 32 64 --mixing moe poe --seed 1 2 3 
```

The script will make 18 configs (2 models x 3 seeds x 3 latent dimensionalities) within the chosen directory. To see the full 
spectrum of parameters that can be adjusted, run:

```python data_proc/generate_configs.py -h```

To automatically run the whole set of experiments located in one folder, launch:

```./iterate_configs.sh "./configs/my_experiment/" ```

We provide sets of configs for the experiments reported in the paper. These are located in _./configs/reproduce_paper/batch_size_experiment_
and  _./configs/reproduce_paper/latent_dim_experiment_. You can run any subset of these using the same bash script. E.g., to reproduce all reported experiments,
run:

```./iterate_configs.sh "./configs/reproduce_paper/" ```
(This is 550 experiments)

Or, to reproduce for example only the latent dimensionality experiments for GeBiD level 5 and the MMVAE model, run:

```./iterate_configs.sh "./configs/reproduce_paper/latent_dim_experiment/gebidlevel5/mmvae"```
(This is 40 experiments)

## Evaluation

After training, you will find various visualizations of the training progress in the _./visuals_ folder of your experiment.
Furthermore, to calculate the joint- and cross-generation accuracy, you can run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python eval/eval_gebid.py --model model_dir_name --level 2  # specify the level on which the model was trained
```

The trained model is expected to be placed in the results folder. The script will print the statistics in the terminal 
and also save them in the model folder as gebid_stats.txt 

You can also view the tensorboard logs by running:

```tensorboard --logdir path_to_model_dir```

Then CTRL + click on the localhost address. If you wish to compare multiple models, put them in one parent directory and provide path to it instead.


## GeBiD leaderboard

Here we show a leaderboard of the state-of-the-art models evaluated on our GeBiD benchmark dataset. The experiments can be 
reproduced by running the configs specified in the Config column (those are linked to a corresponding subfoder in ./configs/reproduce_paper which contains the 5 seeds). For example, to reproduce the leaderboard results
for GeBiD Level 1 and the MVAE model, run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
./iterate_configs.sh "./configs/reproduce_paper/latent_dim_experiment/gebidlevel1/mvae/64/bce/"
```
 
 All experiments will be run with 5 seeds, the results here are reported as a mean over those seeds.
 Here is a legend for the leaderboard tables:
 
 - **Pos.** - position of the model in the leaderboard
 - **Model** - refers to the multimodal VAE model shortcut (e.g. MMVAE, MVAE). 
 - **Obj.** - objective function used for training (ELBO, IWAE, DREG) 
  - **Accuracy (Text&rarr;Image)** - provided only text on the input, we report accuracy of the reconstructed images.
 We show **two** numbers: 
     - **Strict** - percentage of completely correct samples (out of 500 test samples)
     - **Feats** - mean percentage of correct features per sample (for Level 1 same as Strict)
 - **Accuracy (Image&rarr;Text)** - provided only images on the input, we report accuracy of the reconstructed text.
 We show **three** numbers: 
     - **Strict** - percentage of completely correct samples (out of 500 test samples)
     - **Feats** - mean percentage of correct words per sample (for Level 1 same as Strict)
     - **Letters** - mean percentage of correct letters per sample
 - **Accuracy Joint** -  we sample _N_ x 20 (_N_ is the Latent Dim) random vectors from the latent space and reconstruct both text and image. We report **two** numbers:
     - **Strict** - percentage of completely correct and matching samples (out of 500 test samples)
     - **Feats** - mean percentage of correct features (matching for image and text) per sample (for Level 1 same as Strict)
 were matching in the image and text outputs) 
 - **Weights** - download the pretrained weights
 - **Config** - config to reproduce the results

Please note that we are currently preparing weights compatible with the newly-added Pytorch Lightning framework. For evaluating the models using the weights provided below, please checkout the following revision: abd4071da1c034b6496f98e2ff379a92f0b92cde
 
 
In brackets we show standard deviations over the 5 seeds.

### Level 1

<table>
  <tr> 
    <td style="text-align:center "><b>Pos.</b></td>
    <td style="text-align:center"><b>Model</b></td>
    <td style="text-align:center"><b>Obj.</b></td>
    <td  colspan="2" style="text-align:right"><b>Accuracy (Txt&rarr;Img) [%]</b></td>
    <td style="text-align:center" colspan="3"><b>Accuracy (Img&rarr;Txt) [%]</b></td>
    <td style="text-align:center" colspan="2"><b>Joint Accuracy [%]</b></td>
    <td style="text-align:center"><b>Weights</b></td>
   <td style="text-align:center"><b>Config</b></td>
  </tr>
  <tr>
    <td> </td><td> </td><td> </td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Letters</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td> </td> <td> </td> 
    </tr>
  <tr>
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">97 (1)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">16 (1)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">25 (1)</td><td style="text-align:center;white-space:nowrap">7 (1)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/1/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel1/mvae/64/bce">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center">18 (6)</td><td style="text-align:center ">N/A</td><td style="text-align:center ">0 (1)</td><td style="text-align:center ">N/A</td><td style="text-align:center ">16 (5)</td><td style="text-align:center ">21 (7)</td><td style="text-align:center ">N/A</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/1/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel1/mmvae/64/bce">Link</a></td>

</table>


### Level 2
<table>
  <tr> 
    <td style="text-align:center "><b>Pos.</b></td>
    <td style="text-align:center"><b>Model</b></td>
    <td style="text-align:center"><b>Obj.</b></td>
    <td  colspan="2" style="text-align:right"><b>Accuracy (Txt&rarr;Img) [%]</b></td>
    <td style="text-align:center" colspan="3"><b>Accuracy (Img&rarr;Txt) [%]</b></td>
    <td style="text-align:center" colspan="2"><b>Joint Accuracy [%]</b></td>
    <td style="text-align:center"><b>Weights</b></td>
   <td style="text-align:center"><b>Config</b></td>
  </tr>
  <tr>
    <td> </td><td> </td><td> </td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Letters</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td> </td> <td> </td> 
    </tr>
  <tr>
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">70 (0)</td><td style="text-align:center;white-space:nowrap">84 (0)</td><td style="text-align:center;white-space:nowrap">7 (0)</td><td style="text-align:center;white-space:nowrap">27 (2)</td><td style="text-align:center;white-space:nowrap">41 (3)</td><td style="text-align:center;white-space:nowrap">6 (1)</td><td style="text-align:center;white-space:nowrap">49 (0)</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel2/mvae/32/bce">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center">6 (2)</td><td style="text-align:center ">28 (6)</td><td style="text-align:center ">6 (2)</td><td style="text-align:center ">32 (2)</td><td style="text-align:center ">43 (1)</td><td style="text-align:center ">3 (1)</td><td style="text-align:center ">20 (3)</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/2/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel2/mmvae/128/bce">Link</a></td>

</table>


### Level 3

<table>
  <tr> 
    <td style="text-align:center "><b>Pos.</b></td>
    <td style="text-align:center"><b>Model</b></td>
    <td style="text-align:center"><b>Obj.</b></td>
    <td  colspan="2" style="text-align:right"><b>Accuracy (Txt&rarr;Img) [%]</b></td>
    <td style="text-align:center" colspan="3"><b>Accuracy (Img&rarr;Txt) [%]</b></td>
    <td style="text-align:center" colspan="2"><b>Joint Accuracy [%]</b></td>
    <td style="text-align:center"><b>Weights</b></td>
   <td style="text-align:center"><b>Config</b></td>
  </tr>
  <tr>
    <td> </td><td> </td><td> </td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Letters</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td> </td> <td> </td> 
    </tr>
  <tr>
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">49 (2)</td><td style="text-align:center;white-space:nowrap">72 (1)</td><td style="text-align:center;white-space:nowrap">0 (1)</td><td style="text-align:center;white-space:nowrap">20 (1)</td><td style="text-align:center;white-space:nowrap">27 (0)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center;white-space:nowrap">31 (1)</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/3/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel3/mvae/128/bce">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center">0 (0)</td><td style="text-align:center ">22 (1)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center ">12 (4)</td><td style="text-align:center ">26 (3)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center ">12 (2)</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/3/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel3/mmvae/128/bce">Link</a></td>

</table>


### Level 4
<table>
  <tr> 
    <td style="text-align:center "><b>Pos.</b></td>
    <td style="text-align:center"><b>Model</b></td>
    <td style="text-align:center"><b>Obj.</b></td>
    <td  colspan="2" style="text-align:right"><b>Accuracy (Txt&rarr;Img) [%]</b></td>
    <td style="text-align:center" colspan="3"><b>Accuracy (Img&rarr;Txt) [%]</b></td>
    <td style="text-align:center" colspan="2"><b>Joint Accuracy [%]</b></td>
    <td style="text-align:center"><b>Weights</b></td>
   <td style="text-align:center"><b>Config</b></td>
  </tr>
  <tr>
    <td> </td><td> </td><td> </td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Letters</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td> </td> <td> </td> 
    </tr>
  <tr>
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">1 (1)</td><td style="text-align:center;white-space:nowrap">38 (5)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center ">24 (1)</td><td style="text-align:center;white-space:nowrap">26 (2)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/4/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel4/mvae/128/bce">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center">0 (0)</td><td style="text-align:center ">23 (8)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center ">8 (7)</td><td style="text-align:center ">20 (1)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center ">7 (10)</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/4/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel4/mmvae/128/bce">Link</a></td>

</table>


### Level 5

<table>
  <tr> 
    <td style="text-align:center "><b>Pos.</b></td>
    <td style="text-align:center"><b>Model</b></td>
    <td style="text-align:center"><b>Obj.</b></td>
    <td  colspan="2" style="text-align:right"><b>Accuracy (Txt&rarr;Img) [%]</b></td>
    <td style="text-align:center" colspan="3"><b>Accuracy (Img&rarr;Txt) [%]</b></td>
    <td style="text-align:center" colspan="2"><b>Joint Accuracy [%]</b></td>
    <td style="text-align:center"><b>Weights</b></td>
   <td style="text-align:center"><b>Config</b></td>
  </tr>
  <tr>
    <td> </td><td> </td><td> </td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td><b>Letters</b></td><td><b>Strict</b></td><td><b>Feats</b></td><td> </td> <td> </td> 
    </tr>
  <tr>
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">1 (0)</td><td style="text-align:center;white-space:nowrap">35 (1)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">8 (0)</td><td style="text-align:center;white-space:nowrap">20 (1)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">2 (0)</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/5/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel5/mvae/32/bce">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center">0 (0)</td><td style="text-align:center ">16 (2)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center ">15 (2)</td><td style="text-align:center ">18 (1)</td><td style="text-align:center ">0 (0)</td><td style="text-align:center ">9 (2)</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/weights/5/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/latent_dim_experiment/gebidlevel5/mmvae/64/bce">Link</a></td>

</table>



Please feel free to propose your own model and training config so that we can add the results in these tables. 



### Training on other datasets

By default, we also support training on MNIST_SVHN (or MNIST/SVHN only), Caltech-UCSD Birds 200 (CUB) dataset as 
used in the [MMVAE paper](https://arxiv.org/pdf/1911.03393.pdf) and Sprites (as in [this repository](https://github.com/YingzhenLi/Sprites)). We provide the default training configs which
 you can adjust according to your needs (e.g. change the model, loss objective etc.). 


#### MNIST_SVHN

First download the dataset (30 MB in total) before the training. You can run the following:

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/mnist_svhn.zip   # download mnist_svhn dataset
unzip mnist_svhn.zip -d ./data/
python main.py --cfg ./configs/config_mnistsvhn.yml
```

#### CUB

We provide our preprocessed and cleaned version of the dataset (106 MB in total). To download and train, run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/cub.zip   # download CUB dataset
unzip cub.zip -d ./data/
python main.py --cfg ./configs/config_cub.yml
```

#### Sprites

![Sprites](https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/sprites.gif)

You can download the sorted version (4.6 GB) with 3 modalities (image sequences, actions and attributes) and train:

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/sprites.zip   # download Sprites dataset
unzip sprites.zip -d ./data/
python main.py --cfg ./configs/config_sprites.yml
```


[How to train on your own dataset](https://gabinsane.github.io/multimodal-vae-comparison/docs/html/tutorials/adddataset.html)





## Extending for own models and networks

The toolkit is designed so that it enables easy extension for new models, objectives, datasets or encoder/decoder networks. 
<div style="text-align: left">
 <img align="right" src="https://data.ciirc.cvut.cz/public/groups/incognite/GeBiD/uml3.png" width="300"  alt="UML class diagram"/>
</div>

Here you can see the UML diagram of the framework. The toolkit uses the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework which enables automatic separation of the data, models and the training process. A new model (see NewMMVAE in the diagram) can be added as a new class derived from TorchMMVAE. The model constructor will automatically create a BaseVAE class instance for each modality defined in the config - these BaseVAE classes will handle the modality-dependent operations such as encoding and decoding the data, sampling etc. The NewMMVAE class thus only requires the mixing method which defines how the individual posteriors should be mixed, although it is as well possible to change the whole forward pass if needed. 

[Step-by-step tutorial on how to add a new model](https://gabinsane.github.io/multimodal-vae-comparison/docs/html/tutorials/addmodel.html)

New encoder and decoder networks can be added in the corresponding scripts (encoders.py, decoders.py). For choosing these networks in the config,
use only the part of the class name following after the underscore (e.g. CNN for the class Enc_**CNN**). 

## Unit tests

We provide a set of unit tests to check whether any newly-added implementations disrupt any of the existing functions. To run the unit test proceed as follows:

```
cd ~/multimodal-vae-comparison/
py.test .
```

## Common installation problems

For some env configurations, the training might fail on the following:

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/user/miniconda3/envs/multivae/lib/python3.8/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, minimal, minimalegl, offscreen, vnc, webgl.

Aborted (core dumped)
```

In that case, try reinstalling opencv:

```pip uninstall opencv-python```

```pip install opencv-python-headless```

If your torch version does not see CUDA (```print(torch.cuda.is_available())``` is False), try installing pytorch specifically for your CUDA toolkit version, e.g.:

```mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia```


## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  


If you use our toolkit or dataset in your work, please, give us an attribution using the following citation:
```
@misc{https://doi.org/10.48550/arxiv.2209.03048,
  doi = {10.48550/ARXIV.2209.03048},
  url = {https://arxiv.org/abs/2209.03048},  
  author = {Sejnova, Gabriela and Vavrecka, Michal and Stepanova, Karla},  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  title = {Benchmarking Multimodal Variational Autoencoders: GeBiD Dataset and Toolkit},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```


## Acknowledgment

The toolkit features models and functions from the official implementations of [MVAE](https://github.com/mhw32/multimodal-vae-public) ([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) ([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) ([paper](https://openreview.net/forum?id=5Y21V0RDBV)).

## Contact

For any additional questions, feel free to email [sejnogab@fel.cvut.cz](mailto:sejnogab@fel.cvut.cz) 
