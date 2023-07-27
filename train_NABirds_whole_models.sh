#!/usr/bin/env bash

# NABirds
# method: Softmax, NormFace, ProxyDR, CORR
# Seeds: 1, 2, 3, 4, 5


# Seed: 1
python train_nabirds.py --GPU 0 --method softmax --distance euc --seed 1 --use_val --aug --resize 128 

python train_nabirds.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --resize 128 
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --resize 128 --mds_W --CORR



# Seed: 2
python train_nabirds.py --GPU 0 --method softmax --distance euc --seed 2 --use_val --aug --resize 128 

python train_nabirds.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --resize 128 
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --resize 128 --mds_W --CORR



# Seed: 3
python train_nabirds.py --GPU 0 --method softmax --distance euc --seed 3 --use_val --aug --resize 128 

python train_nabirds.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --resize 128 
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --resize 128 --mds_W --CORR



# Seed: 4
python train_nabirds.py --GPU 0 --method softmax --distance euc --seed 4 --use_val --aug --resize 128 

python train_nabirds.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --resize 128 
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --resize 128 --mds_W --CORR



# Seed: 5
python train_nabirds.py --GPU 0 --method softmax --distance euc --seed 5 --use_val --aug --resize 128 

python train_nabirds.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --resize 128 
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128 --ema
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128 --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128 --ema --dynamic
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128 --mds_W
python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128 --mds_W --dynamic

python train_nabirds.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --resize 128 --mds_W --CORR


