#!/usr/bin/env bash

# MesoZ
# method: Softmax, NormFace, ProxyDR, CORR
# Seeds: 1, 2, 3, 4, 5

# Seed: 1
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 1 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 1 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 1 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 1 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 1 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 1 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 1 --use_val --mds_W --CORR
python eval_.py --GPU 0 --dataset MesoZ --method softmax --distance euc --size_inform --seed 1 --use_val

# Seed: 2
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 2 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 2 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 2 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 2 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 2 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 2 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 2 --use_val --mds_W --CORR
python eval_.py --GPU 0 --dataset MesoZ --method softmax --distance euc --size_inform --seed 2 --use_val

# Seed: 3
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 3 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 3 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 3 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 3 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 3 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 3 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 3 --use_val --mds_W --CORR
python eval_.py --GPU 0 --dataset MesoZ --method softmax --distance euc --size_inform --seed 3 --use_val

# Seed: 4
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 4 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 4 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 4 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 4 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 4 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 4 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 4 --use_val --mds_W --CORR
python eval_.py --GPU 0 --dataset MesoZ --method softmax --distance euc --size_inform --seed 4 --use_val

# Seed: 5
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 5 --use_val
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 5 --use_val --ema
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 5 --use_val --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 5 --use_val --ema --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 5 --use_val --mds_W
python eval_.py --GPU 0 --dataset MesoZ --method normface --distance euc --size_inform --seed 5 --use_val --mds_W --dynamic
python eval_.py --GPU 0 --dataset MesoZ --method DR --distance euc --size_inform --seed 5 --use_val --mds_W --CORR
python eval_.py --GPU 0 --dataset MesoZ --method softmax --distance euc --size_inform --seed 5 --use_val
