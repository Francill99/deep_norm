>📋  A template README.md for code accompanying a Machine Learning paper

# Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks

This repository is the official implementation of (https://arxiv.org/abs/2505.13230). 

## Requirements

To install requirements:

```setup
conda env create --file deep_norm-environment.yml
conda activate deep_norm
```

To download datasets:
python download_datasets.pbs

## Training

To train a single run of a model and dataset used in the paper, run this command:

```train
python CNN_train.py --P 48000 --T 500 --lr 1e-3 --seed 11130 --dataset CIFAR10 --data_root ./data --out_dir <select_directory> 
```
To obtain the curves in the paper it is necessary to run for different P values and seed choices.
