<h1 align="center">DDX7: DDX7: Differentiable FM Synthesis of Musical Instrument Sounds</h1>
<div align="center">
<h4>
    <a href="" target="_blank">paper</a> - <a href="https://fcaspe.github.io/ddx7" target="_blank">website</a>
</h4>
    <p>
    Franco Caspe - Andrew McPherson - Mark Sandler
    </p>
</div>

This is the official implementation of the paper, accepted to the 23rd International Society
for Music Information Retrieval Conference [ISMIR 2022](https://ismir2022.ismir.net/).

## Install

It is reccomended to install this repo on a virtual environment.


    pip install -r requirements.txt
    pip install -e .

Also make sure `pytorch` is setup with the [CUDA version](https://pytorch.org/get-started/locally/)
that support the capabilities of your GPU.

## About configuration

This codebase employs [`Hydra`](https://hydra.cc/) to personalize dataset generation, and build and train models.
Please checkout available options in `yaml` files before processing a dataset or training a model.

## Dataset Generation

We used the [URMP](https://labsites.rochester.edu/air/projects/URMP.html) dataset to train and test the models.
Additional test files can be aggregated and used for resynthesis tasks.
Please check the `dataset` directory for advanced options to process and build a dataset.

**Quick start** - will extract and process violin, flute, and trumpet data with [`torchcrepe`](https://github.com/maxrmorrison/torchcrepe).

    python dataset/create_data.py urmp.source_folder=/path/to/URMP/Dataset


## Training

Please check the `recipes` directory for available models and hyperparameters.

**Quick start: train DDX7** - will train a DDX7 model on violin data on GPU.

    python train.py # override GPU with "device=cpu" option.


## Citation
