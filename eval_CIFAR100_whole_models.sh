#!/usr/bin/env bash

# CIFAR-100
# method: Softmax, NormFace, ProxyDR, CORR
# Seeds: 1, 2, 3, 4, 5

# Seed: 1
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug
python eval_cifar100.py --GPU 0 --method softmax --distance euc --seed 1 --use_val --aug
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 1 --use_val --aug --ema --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 1 --use_val --aug --ema --dynamic

# Seed: 2
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug
python eval_cifar100.py --GPU 0 --method softmax --distance euc --seed 2 --use_val --aug
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 2 --use_val --aug --ema --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 2 --use_val --aug --ema --dynamic

# Seed: 3
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug
python eval_cifar100.py --GPU 0 --method softmax --distance euc --seed 3 --use_val --aug
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 3 --use_val --aug --ema --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 3 --use_val --aug --ema --dynamic

# Seed: 4
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug
python eval_cifar100.py --GPU 0 --method softmax --distance euc --seed 4 --use_val --aug
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 4 --use_val --aug --ema --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 4 --use_val --aug --ema --dynamic

# Seed: 5
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug
python eval_cifar100.py --GPU 0 --method softmax --distance euc --seed 5 --use_val --aug
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --ema
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --dynamic
python eval_cifar100.py --GPU 0 --method DR --distance euc --seed 5 --use_val --aug --ema --dynamic
python eval_cifar100.py --GPU 0 --method normface --distance euc --seed 5 --use_val --aug --ema --dynamic



