# Multimodal VAE Comparison

This is the official code for the submitted NIPS 2023 Datasets and Benchmarks paper "Benchmarking Multimodal Variational Autoencoders: CdSprites+ Dataset and Toolkit".

The purpose of this toolkit is to offer a systematic and unified way to train, evaluate and compare the state-of-the-art
multimodal variational autoencoders. The toolkit can be used with arbitrary datasets and both uni/multimodal settings.
By default, we provide implementations of the [MVAE](https://github.com/mhw32/multimodal-vae-public) 
([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) 
([paper](https://arxiv.org/pdf/1911.03393.pdf)), [MoPoE](https://github.com/thomassutter/MoPoE) 
([paper](https://openreview.net/forum?id=5Y21V0RDBV)) and [DMVAE](https://github.com/seqam-lab/DMVAE) ([paper](https://github.com/seqam-lab/DMVAE)) models, but anyone is free to contribute with their own
implementation. 

We also provide a custom synthetic bimodal dataset, called **CdSprites+**, designed specifically for comparison of the
joint- and cross-generative capabilities of multimodal VAEs. You can read about the utilities of the dataset in the proposed 
paper (link will be added soon). This dataset extends the [dSprites dataset](https://github.com/deepmind/dsprites-dataset) with natural language captions and additional features and offers 5 levels of difficulty (based on the number of attributes)
to find the minimal functioning scenario for each model. Moreover, its rigid structure enables automatic qualitative
evaluation of the generated samples. For more info, see below. 

[**Code Documentation & Tutorials**](https://gabinsane.github.io/multimodal-vae-comparison)


| :dart: To elevate the general discussion on the development and evaluation of multimodal VAEs, we have now added the [Discussions section](https://github.com/gabinsane/multimodal-vae-comparison/discussions) |
| --- |

---
### **List of contents**

* [Preliminaries](#preliminaries) <br>
* [CdSprites+ dataset](#get-the-cdsprites-dataset) <br>
* [Setup & Training](#setup-and-training) <br>
* [Evaluation](#evaluation)<br>
* [CdSprites+ leaderboard](#cdsprites&#43;-leaderboard)<br>
* [Training on other datasets](#training-on-other-datasets) <br>
* [Add own model](#extending-for-own-models-and-networks)<br>
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


## Get the CdSprites&#43; dataset

We provide a bimodal image-text dataset CdSprites+ (Geometric shapes Bimodal Dataset) for systematic multimodal VAE comparison. There are 5 difficulty levels 
based on the number of featured attributes (shape, size, color, position and background color). You can either generate
the dataset on your own, or download a ready-to-go version.

### Dataset download 
You can download any of the following difficulty levels: [Level 1](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/level1.zip),
[Level 2](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/level2.zip), [Level 3](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/level3.zip),
[Level 4](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/level4.zip), [Level 5](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/level5.zip).

The dataset should be placed in the ./data/CdSpritesplus directory. For downloading, unzipping and moving the chosen dataset, run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/level2.zip   # replace level2 with any of the 1-5 levels
unzip level2.zip -d ./data/CdSpritesplus
```

![Examples of CdSprites+ levels](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/cdsprites.png "CdSprites+ dataset")

### Dataset generation

You can also generate the dataset on your own. To generate all levels at once, run:

 ```
cd ~/multimodal-vae-comparison/multimodal_compare/data_proc
python ./cdSprites.py 
```

Alternatively, to generate only one level:

```
cd ~/multimodal-vae-comparison/multimodal_compare/data_proc
python ./cdSprites.py --level 4
```

The code will create the _./CdSpritesplus_ folder in the _./data_ directory. The folder includes subfolders with different levels (or the one level that you have chosen). Each level contains images sorted in directories according to their captions. There is also the _traindata.h5_ file containing the whole dataset - you can then use this file in the config.

## Setup and training

### Single experiment
We show an example training config in _./multimodal_compare/configs/config1.yml_. You can run the training as follows (assuming you downloaded or generated the dataset level 2 above):

```
cd ~/multimodal-vae-comparison/multimodal_compare
python main.py --cfg configs/config1.yml
```

The config contains general arguments and modality-specific arguments (denoted as "modality_n"). In general, you can set up a training for 1-N modalities by defining the required subsections for each of them. 
The paths to all modalities are expected to have the data ordered so that they are semantically matching (e.g. the first image and the first text sample belong together).

The usage and possible options for all the config arguments are below:

![Config documentation](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/config2.png "config documentation")

### Set of experiments

We provide an automated way to perform a hyperparameter grid search for your models. First, set up the default config (e.g. _config1.yml_ in _./configs_)
that should be adjusted in the selected parameters. Then generate the full variability within the chosen parameters as follows:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python data_proc/generate_configs.py --path ./configs/my_experiment  --cfg ./configs/config1.yml --n-latents 24 32 64 --mixing moe poe --seed 1 2 3 
```

The script will make 36 configs (4 models x 3 seeds x 3 latent dimensionalities) within the chosen directory. To see the full 
spectrum of parameters that can be adjusted, run:

```python data_proc/generate_configs.py -h```

To automatically run the whole set of experiments located in one folder, launch:

```./iterate_configs.sh "./configs/my_experiment/" ```

We provide sets of configs for the experiments reported in the paper. These are located in _./configs/reproduce_paper/_. You can run any subset of these using the same bash script. E.g., to reproduce all reported experiments,
run:

```./iterate_configs.sh "./configs/reproduce_paper/" ```
(This is 60 experiments, each run trains the model 5x with 5 different seeds.)

Or, to reproduce for example only the experiments for the MMVAE model, run:

```./iterate_configs.sh "./configs/reproduce_paper/mmvae"```
(This is 15 experiments, each run trains the model 5x with 5 different seeds.)

### Mixed precision training

For improving the training speed, you can also use Mixed Precision Training. PyTorch Lightning supports the following values: **64**, **32**, **16**, **bf16**. 
The default precision is 32, but you can change the parameter with the '--precision' or '-p' argument:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python main.py --cfg configs/config1.yml --precision bf16
```

You can read more about this configuration in the [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/1.5.7/advanced/mixed_precision.html)

## Evaluation

After training, you will find various visualizations of the training progress in the _./visuals_ folder of your experiment.
Furthermore, to calculate the joint- and cross-generation accuracy, you can run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python eval/eval_cdsprites.py --model model_dir_name --level 2  # specify the level on which the model was trained
```

The trained model is expected to be placed in the results folder. The script will print the statistics in the terminal 
and also save them in the model folder as cdsprites_stats.txt 


## CdSprites&#43; leaderboard

Here we show a leaderboard of the state-of-the-art models evaluated on our CdSprites+ benchmark dataset. The experiments can be 
reproduced by running the configs specified in the Config column (those are linked to a corresponding subfoder in ./configs/reproduce_paper which contains the 5 seeds). For example, to reproduce the leaderboard results
for CdSprites+ Level 1 and the MVAE model, run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
./iterate_configs.sh "./configs/reproduce_paper/mvae/level1"
```
 
 All experiments will be run with 5 seeds, the results here are reported as a mean over those seeds.
 Here is a legend for the leaderboard tables:
 
 - **Pos.** - position of the model in the leaderboard
 - **Model** - refers to the multimodal VAE model shortcut (e.g. MMVAE, MVAE). 
 - **Obj.** - objective function used for training (ELBO, IWAE, DREG) 
  - **Accuracy (Text&rarr;Image)** - provided only text on the input, we report accuracy of the reconstructed images.
 We show **two** numbers: 
     - **Strict** - percentage of completely correct samples (out of 500 test samples)
     - **Feats** - ratio of correct features per sample, i.e., 1.2 (0.1)/3 for Level 3 means that on average 1.2 +/- 0.1 features out of 3 are recognized correctly for each sample (for Level 1 same as Strict)
 - **Accuracy (Image&rarr;Text)** - provided only images on the input, we report accuracy of the reconstructed text.
 We show **three** numbers: 
     - **Strict** - percentage of completely correct samples (out of 250 test samples)
     - **Feats** - ratio of correct words per sample (for Level 1 same as Strict)
     - **Letters** - mean percentage of correct letters per sample
 - **Accuracy Joint** -  we sample _N_ x 20 (_N_ is the Latent Dim) random vectors from the latent space and reconstruct both text and image. We report **two** numbers:
     - **Strict** - percentage of completely correct and matching samples (out of 500 test samples)
     - **Feats** - ratio of correct features (matching for image and text) per sample (for Level 1 same as Strict)
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
     <td style="text-align:center ">1.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center">47(14)</td><td style="text-align:center ">N/A</td><td style="text-align:center ">64 (3)</td><td style="text-align:center ">N/A</td><td style="text-align:center ">88 (2)</td><td style="text-align:center ">17 (10)</td><td style="text-align:center ">N/A</td><td style="text-align:center "><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/1/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mmvae/level1">Link</a></td>
  </tr>
        <td style="text-align:center ">2.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">52 (3)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">63 (8)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">86 (2)</td><td style="text-align:center;white-space:nowrap">5 (9)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/1/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mvae/level1">Link</a></td> 
</tr>
        <td style="text-align:center ">3.</td> <td style="text-align:center ">MoPoE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">33 (3)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">10 (17)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">26 (7)</td><td style="text-align:center;white-space:nowrap">16 (27)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/1/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mopoe/level1">Link</a></td> 
</tr>
        <td style="text-align:center ">4.</td> <td style="text-align:center ">DMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">33 (4)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">4 (5)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap">25 (2)</td><td style="text-align:center;white-space:nowrap">4 (6)</td><td style="text-align:center;white-space:nowrap">N/A</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/1/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/dmvae/level1">Link</a></td> 

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
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">16 (1)</td><td style="text-align:center;white-space:nowrap">0.8 (0.0)/2</td><td style="text-align:center;white-space:nowrap">55 (27)</td><td style="text-align:center;white-space:nowrap">1.5 (0.3)/2</td><td style="text-align:center;white-space:nowrap">91 (6)</td><td style="text-align:center;white-space:nowrap">1 (1)</td><td style="text-align:center;white-space:nowrap">0.3 (0.3)/2</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mvae/level2">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">18 (4)</td><td style="text-align:center;white-space:nowrap">0.8 (0.1)/2</td><td style="text-align:center;white-space:nowrap">41 (20)</td><td style="text-align:center;white-space:nowrap">1.4 (0.2)/2</td><td style="text-align:center;white-space:nowrap">85 (4)</td><td style="text-align:center;white-space:nowrap">3 (3)</td><td style="text-align:center;white-space:nowrap">0.6 (0.1)/2</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mmvae/level2">Link</a></td>
  </tr>
         <td style="text-align:center ">3.</td> <td style="text-align:center ">MoPoE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">10 (3)</td><td style="text-align:center;white-space:nowrap">0.8 (0.0)/2</td><td style="text-align:center;white-space:nowrap">8 (7)</td><td style="text-align:center;white-space:nowrap">0.7 (0.1)/2</td><td style="text-align:center;white-space:nowrap">40 (4)</td><td style="text-align:center;white-space:nowrap">1 (1)</td><td style="text-align:center;white-space:nowrap">0.2 (0.1)/2</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mopoe/level2">Link</a></td>
  </tr>
         <td style="text-align:center ">4.</td> <td style="text-align:center ">DMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">15 (2)</td><td style="text-align:center;white-space:nowrap">0.8 (0.0)/2</td><td style="text-align:center;white-space:nowrap">4 (1)</td><td style="text-align:center;white-space:nowrap">0.4 (0.0)/2</td><td style="text-align:center;white-space:nowrap">30 (2)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.2 (0.1)/2</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/dmvae/level2">Link</a></td>
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
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">8 (2)</td><td style="text-align:center;white-space:nowrap">1.3 (0.0)/3</td><td style="text-align:center;white-space:nowrap">59 (4)</td><td style="text-align:center;white-space:nowrap">2.5 (0.3)/3</td><td style="text-align:center;white-space:nowrap">93 (1)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.5 (0.1)/3</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mvae/level3">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">6 (2)</td><td style="text-align:center;white-space:nowrap">1.2 (0.2)/3</td><td style="text-align:center;white-space:nowrap">2 (3)</td><td style="text-align:center;white-space:nowrap">0.6 (0.2)/3</td><td style="text-align:center;white-space:nowrap">31 (5)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.4 (0.1)/3</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mmvae/level3">Link</a></td>
    </tr>
         <td style="text-align:center ">3.</td> <td style="text-align:center ">MoPoE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">7 (4)</td><td style="text-align:center;white-space:nowrap">1.3 (0.1)/3</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.7 (0.1)/3</td><td style="text-align:center;white-space:nowrap">32 (0)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.1 (0.1)/3</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mopoe/level3">Link</a></td>
  </tr>
         <td style="text-align:center ">4.</td> <td style="text-align:center ">DMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">4 (0)</td><td style="text-align:center;white-space:nowrap">1.4 (0.0)/3</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.4 (0.1)/3</td><td style="text-align:center;white-space:nowrap">22 (2)</td><td style="text-align:center;white-space:nowrap">1 (1)</td><td style="text-align:center;white-space:nowrap">0.5 (0.1)/3</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/2/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/dmvae/level3">Link</a></td>

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
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.8 (0.0)/4</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center ">0.6 (0.0)/4</td><td style="text-align:center;white-space:nowrap">28 (3)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.6 (0.0)/4</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/4/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mvae/level4">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">3 (3)</td><td style="text-align:center;white-space:nowrap">1.7 (0.4)/4</td><td style="text-align:center;white-space:nowrap">1 (2)</td><td style="text-align:center ">0.7 (0.4)/4</td><td style="text-align:center;white-space:nowrap">27 (9)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.5 (0.2)/4</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/4/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mvae/level4">Link</a></td>
   </tr>
         <td style="text-align:center ">3.</td> <td style="text-align:center ">MoPoE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">2 (1)</td><td style="text-align:center;white-space:nowrap">1.4 (0.0)/4</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center ">0.7 (0.1)/4</td><td style="text-align:center;white-space:nowrap">21 (3)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.1 (0.2)/4</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/4/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mopoe/level4">Link</a></td>
    </tr>
         <td style="text-align:center ">4.</td> <td style="text-align:center ">DMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">1 (1)</td><td style="text-align:center;white-space:nowrap">1.4 (0.0)/4</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center ">0.5 (0.1)/4</td><td style="text-align:center;white-space:nowrap">18 (1)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.5 (0.1)/4</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/4/mmvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/dmvae/level4">Link</a></td>

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
       <td style="text-align:center ">1.</td> <td style="text-align:center ">MVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.8 (0.0)/5</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.6 (0.0)/5</td><td style="text-align:center;white-space:nowrap">27 (1)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.2 (0.2)/5</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/5/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mvae/level5">Link</a></td>
  </tr>
         <td style="text-align:center ">2.</td> <td style="text-align:center ">MMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.8 (0.0)/5</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.1 (0.1)/5</td><td style="text-align:center;white-space:nowrap">13 (2)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.4 (0.1)/5</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/5/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mmvae/level5">Link</a></td>
   </tr>
         <td style="text-align:center ">3.</td> <td style="text-align:center ">MoPoE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.8 (0.0)/5</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.7 (0.0)/5</td><td style="text-align:center;white-space:nowrap">17 (1)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.0 (0.0)/5</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/5/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mopoe/level5">Link</a></td>
    </tr>
         <td style="text-align:center ">4.</td> <td style="text-align:center ">DMVAE</td><td>ELBO</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">1.8 (0.0)/5</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.6 (0.1)/5</td><td style="text-align:center;white-space:nowrap">18 (2)</td><td style="text-align:center;white-space:nowrap">0 (0)</td><td style="text-align:center;white-space:nowrap">0.7 (0.1)/5</td><td style="text-align:center;white-space:nowrap"><a href="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/weights/5/mvae.zip">Link</a></td><td style="text-align:center;white-space:nowrap"><a href="https://github.com/gabinsane/multimodal-vae-comparison/tree/main/multimodal_compare/configs/reproduce_paper/mmvae/level5">Link</a></td>

</table>



Please feel free to propose your own model and training config so that we can add the results in these tables. 



### Training on other datasets

By default, we also support training on MNIST_SVHN (or MNIST/SVHN only), Caltech-UCSD Birds 200 (CUB) dataset as 
used in the [MMVAE paper](https://arxiv.org/pdf/1911.03393.pdf), Sprites (as in [this repository](https://github.com/YingzhenLi/Sprites)), CelebA, FashionMNIST and PolyMNIST. We provide the default training configs which
 you can adjust according to your needs (e.g. change the model, loss objective etc.). 


#### MNIST_SVHN

We use the inbuilt torchvision.datasets function to download and process the dataset. Resampling of the data
should happen automatically based on indices that will be downloaded within the script. You can thus run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python main.py --cfg configs/config_mnistsvhn.yml
```


#### CUB

We provide our preprocessed and cleaned version of the dataset (106 MB in total). To download and train, run:

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/cub.zip   # download CUB dataset
unzip cub.zip -d ./data/
python main.py --cfg configs/config_cub.yml
```

#### Sprites
 
You can download the sorted version (4.6 GB) with 3 modalities (image sequences, actions and attributes) and train:

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/sprites.zip   # download Sprites dataset
unzip sprites.zip -d ./data/
python main.py --cfg configs/config_sprites.yml
```

#### CelebA

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/celeba.zip   # download CelebA dataset
unzip celeba.zip -d ./data/
python main.py --cfg configs/config_celeba.yml
```

#### FashionMNIST

For FashionMNIST, we use the torchvision.datasets class to handle the download automatically, you thus do not need to download anything. 
You can train directly by running:

```
cd ~/multimodal-vae-comparison/multimodal_compare
python main.py --cfg configs/config_fashionmnist.yml
```

#### PolyMNIST

```
cd ~/multimodal-vae-comparison/multimodal_compare
wget https://zenodo.org/record/4899160/files/PolyMNIST.zip?download=1   # download PolyMNIST dataset
unzip PolyMNIST.zip?download=1 -d ./data/
python main.py --cfg configs/config_polymnist.yml
```


[How to train on your own dataset](https://gabinsane.github.io/multimodal-vae-comparison/docs/html/tutorials/adddataset.html)





## Extending for own models and networks

The toolkit is designed so that it enables easy extension for new models, objectives, datasets or encoder/decoder networks. 
<div style="text-align: left">
 <img align="right" src="https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/uml3.png" width="300"  alt="UML class diagram"/>
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


## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  


If you use our toolkit or dataset in your work, please, give us an attribution using the following citation:
```
@misc{https://doi.org/10.48550/arxiv.2209.03048,
  doi = {10.48550/ARXIV.2209.03048},
  url = {https://arxiv.org/abs/2209.03048},  
  author = {Sejnova, Gabriela and Vavrecka, Michal and Stepanova, Karla},  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  title = {Benchmarking Multimodal Variational Autoencoders: CdSprites Dataset and Toolkit},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```


## Acknowledgment

The toolkit features models and functions from the official implementations of [MVAE](https://github.com/mhw32/multimodal-vae-public) ([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) ([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) ([paper](https://openreview.net/forum?id=5Y21V0RDBV)).

## Contact

For any additional questions, feel free to email [sejnogab@fel.cvut.cz](mailto:sejnogab@fel.cvut.cz) 
