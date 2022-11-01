# Inspecting_Hierarchies_ML
Code for paper the "Inspecting class hierarchies of classification-based metric learning models"

## Environment
 - Python3
 - Pytorch (http://pytorch.org/) (version ...)

## Preparing datasets
### Three plankton datasets
Small microplankton (MicroS), Large microplankton (MicroL), Mesozooplankton (MesoZ)...


### CIFAR-100
We used CIFAR-100 from torchvision https://pytorch.org/vision/stable/datasets.html.

One may download CIFAR-100 dataset from https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-100 python version). 


### NABirds
One can download NABirds dataset from https://dl.allaboutbirds.org/nabirds.

## Train
For training of plankton datasets, run `python train.py --GPU [GPU_NUMBER(S)] --dataset [DATASET_NAME] --method [METHODNAME] --distance [DISTANCE] --size_inform --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

## Evaluation
Run `...`

## Results
The training and evaludation results will be recorded in `./record/`

## References 



