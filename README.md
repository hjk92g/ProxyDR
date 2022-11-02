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

For training of CIFAR100 dataset, run `python train_cifar100.py --GPU [GPU_NUMBER(S)] --method [METHODNAME] --distance [DISTANCE] --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

For training of NABird dataset, run `python train_nabirds.py --GPU [GPU_NUMBER(S)] --method [METHODNAME] --distance [DISTANCE] --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

### Methods 
Softmax: , NormFace: , ProxyDR:, CORR loss: 


### Training options and the corresponding [TRAINING_OPTION] names
Standard: default (without any --[TRAINING_OPTION]), EMA: --ema, Dynamic: --dynamic, MDS (multidimensional scaling): --mds_W

### Code examples
For example, to train NormFace model on MicroS dataset with standard option (also GPU:0, seed: 1, use Euclidean distance, size information and validation), run `python train.py --GPU 0 --dataset MicroS --method SD --distance euc --size_inform --seed 1 --use_val`

For example, to train ProxyDR model on MicroS dataset with MDS and dynamic options (also GPU:0, seed: 1, use Euclidean distance, size information and validation), run `python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --dynamic`

## Evaluation
Run `...`

## Results
The training and evaludation results will be recorded in `./record/`

## References 



