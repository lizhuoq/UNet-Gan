#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=UNet

python -u pretrain_era5.py \
    --num_epochs 200 \
    --generator $model_name \
    --in_channel 17 \
    --out_channel 3 \
    --G_learning_rate 5e-4 \
    --D_learning_rate 1e-6 \
    --patience 10 \
    --batch_size 100 \
    --warmup_epochs 0 \
    --perceptual_loss vgg16 \
    --kernel_size 3 \
    --test