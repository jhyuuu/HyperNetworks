#/usr/bin/bash

COMMOM='--model MobileNetV2 --batch_size 128 --epochs 150 --dataset cifar10 --wd 4e-5'


CUDA_VISIBLE_DEVICES=3 python train.py > log/hpnet_run0.log &