#!/bin/bash

module load anaconda/2020.11
module load cuda/11.8
source activate climsim

model_name=UNet

for i in {1..5}
do
    for j in {0..3}
    do
        python -u finetune_amsmqtp.py \
            --num_epochs 250 \
            --generator $model_name \
            --in_channel 17 \
            --out_channel 5 \
            --G_learning_rate 0.00035 \
            --D_learning_rate 7e-7 \
            --patience 10 \
            --batch_size 50 \
            --warmup_epochs 0 \
            --perceptual_loss vgg16 \
            --kernel_size 3 \
            --fold_n $i \
            --point_alpha 0.01 \
            --inference \
            --exp_id $j
    done
done