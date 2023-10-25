#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python ${CUR_DIR}/train.py --dataroot ${CUR_DIR}/datasets/soccer_seg_detection \
    --gpu_ids -1 \
    --name soccer_seg_detection_pix2pix --model two_pix2pix --netG unet_256 \
    --direction AtoB --dataset_mode two_aligned \
    --gan_mode vanilla --norm batch --pool_size 0 --output_nc 1 \
    --phase1 train_phase_1 --phase2 train_phase_2 --save_epoch_freq 2
