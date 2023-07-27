#!/usr/bin/env bash

# Small microplankton (MicroS)
# method: Softmax, NormFace, ProxyDR, CORR
# Seeds: 1, 2, 3, 4, 5


# Seed: 1
python train.py --GPU 0 --dataset MicroS --method softmax --distance euc --size_inform --seed 1 --use_val 

python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 1 --use_val 
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 1 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 1 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 1 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 1 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 1 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val 
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --CORR



# Seed: 2
python train.py --GPU 0 --dataset MicroS --method softmax --distance euc --size_inform --seed 2 --use_val 

python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 2 --use_val 
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 2 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 2 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 2 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 2 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 2 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val 
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 2 --use_val --mds_W --CORR



# Seed: 3
python train.py --GPU 0 --dataset MicroS --method softmax --distance euc --size_inform --seed 3 --use_val 

python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 3 --use_val 
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 3 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 3 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 3 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 3 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 3 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val 
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 3 --use_val --mds_W --CORR



# Seed: 4
python train.py --GPU 0 --dataset MicroS --method softmax --distance euc --size_inform --seed 4 --use_val 

python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 4 --use_val 
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 4 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 4 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 4 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 4 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 4 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val 
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 4 --use_val --mds_W --CORR



# Seed: 5
python train.py --GPU 0 --dataset MicroS --method softmax --distance euc --size_inform --seed 5 --use_val 

python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 5 --use_val 
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 5 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 5 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 5 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 5 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method normface --distance euc --size_inform --seed 5 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val 
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val --ema
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val --ema --dynamic
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val --mds_W
python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val --mds_W --dynamic

python train.py --GPU 0 --dataset MicroS --method DR --distance euc --size_inform --seed 5 --use_val --mds_W --CORR


