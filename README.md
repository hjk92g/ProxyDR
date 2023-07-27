# Inspecting_Hierarchies_ML
Code for the paper ["Inspecting class hierarchies of classification-based metric learning models"](https://arxiv.org/abs/2301.11065)

## Environment
 - Python3
 - PyTorch (http://pytorch.org/) (gpytorch 1.4.1)
 - NumPy (version 1.19.5)
 - Pandas (version 1.0.5)
 - Scikit-learn (version 0.24.2)
 - SciPy (version 1.5.0)
 - Biopython (version 1.79)

## Preparing datasets
### Three plankton datasets
You can download these from [Small microplankton (MicroS)](https://doi.org/10.21335/NMDC-2102309336), [Large microplankton (MicroL)](https://doi.org/10.21335/NMDC-573815973), and [Mesozooplankton (MesoZ)](https://doi.org/10.21335/NMDC-1805578916). These datasets should be inside a folder named "plankton_data" (you need to make this folder). You need to change path names in `MicroS_cls.csv`, `MicroS_info.csv`, `MicroL_cls.csv`, `MicroL_info.csv`, `MesoZ_cls.csv`, and `MesoZ_info.csv` such that images are located in the written path (you will only need to change "DATA_init" to the corresponding folder name in each line. For instance, you might use the command `sed -i 's/DATA_init/Data_path_name/g' MicroS_cls.csv`).

### CIFAR-100
We used CIFAR-100 from torchvision https://pytorch.org/vision/stable/datasets.html.

One may download CIFAR-100 dataset from https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-100 python version). 

### NABirds
One can download NABirds dataset from https://dl.allaboutbirds.org/nabirds. You need to change path names in `nabirds_cls.csv`, `nabirds_cls2.csv`, and `nabirds_info.csv` such that images are located in the written path (you will only need to change "DATA_init" to the corresponding folder name in each line).
You need to run `Prepare_NABirds.ipynb` after properly changing the `config.json` file as explained in the train section.


## Train
Before training, in the `config.json` file, you need to put where the "plankton_data" and "nabirds" folders are located (`DATA_init`) and where this repogistory (`Inspecting_Hierarchies_ML`) is located (`FOLDER_init`).

For training of plankton datasets, run `python train.py --GPU [GPU_NUMBER(S)] --dataset [DATASET_NAME] --method [METHOD_NAME] --distance [DISTANCE] --size_inform --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

For training of CIFAR100 dataset, run `python train_cifar100.py --GPU [GPU_NUMBER(S)] --method [METHOD_NAME] --distance [DISTANCE] --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

For training of NABird dataset, run `python train_nabirds.py --GPU [GPU_NUMBER(S)] --method [METHOD_NAME] --distance [DISTANCE] --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

 - **Methods** (`[METHOD_NAME]`)<br>
 Softmax: `softmax`, NormFace: `normface`, ProxyDR: default `DR`, CORR loss: `--method DR --mds_W --CORR`

 - **Training options and the corresponding `[TRAINING_OPTION]` names** <br>
 Standard: default (without any --[TRAINING_OPTION]), EMA: `--ema`, Dynamic (scale factor): `--dynamic`, MDS (multidimensional scaling): `--mds_W`

### Code examples
For example, to train NormFace model on MicroS dataset with standard option (also GPU:0, seed: 1, use Euclidean distance, size information and validation), run `python train.py --GPU 0 --dataset MicroS --method SD --distance euc --size_inform --seed 1 --use_val`

For example, to train ProxyDR model on MicroS dataset with MDS and dynamic options (also GPU:0, seed: 1, use Euclidean distance, size information and validation), run `python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --dynamic`

For example, to train CORR model (requires MDS) on MicroS dataset  (also GPU:0, seed: 1, use Euclidean distance, size information and validation), run `python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --CORR`

### Training whole models (replicating experiments in our paper)
If you want to replicate the experiments, instead of typing each training setting, you can run `train_MicroS_whole_models.sh`, `train_MicroL_whole_models.sh`, `train_MesoZ_whole_models.sh`, `train_CIFAR100_whole_models.sh`, and `train_NABirds_whole_models.sh`. (You may want to change GPU number. Values might be differ due to randomness.)

## Evaluation of trained models
For evaluation of plankton dataset models, run `python eval_.py --GPU [GPU_NUMBER(S)] --dataset [DATASET_NAME] --method [METHODNAME] --distance [DISTANCE] --size_inform --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

For evaluation of CIFAR100 dataset models, run `python eval_cifar100.py --GPU [GPU_NUMBER(S)] --method [METHOD_NAME] --distance [DISTANCE] --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

For evaluation of NABird dataset models, run `python eval_nabirds.py --GPU [GPU_NUMBER(S)] --method [METHOD_NAME] --distance [DISTANCE] --use_val --seed [SEED_NUMBER] --[TRAINING_OPTION]`.

 - `--last`: evaluate the last training epoch model (probably not the best model)

### Evaluating whole models (replicating experiments in our paper)
If you want to replicate the experiments, instead of typing each evaluation setting, you can run `eval_MicroS_whole_models.sh`, `eval_MicroL_whole_models.sh`, `eval_MesoZ_whole_models.sh`, `eval_CIFAR100_whole_models.sh`, and `eval_NABirds_whole_models.sh`. (You may want to change GPU number. Values might be differ due to randomness.)

## Results
The training and evaludation results will be recorded in `./record/`

## References 
Dynamic option implementation is modified from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py.


