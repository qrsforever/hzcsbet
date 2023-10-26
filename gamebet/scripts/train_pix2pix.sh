#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

python3 ${TOP_DIR}/train_pix2pix.py --dataroot ${TOP_DIR}/datasets/soccer_seg_detection \
    --gpu_ids -1 --phase train_phase_1 --display_id -1 --print_freq 10 \
    --name soccer_seg_detection_pix2pix --model pix2pix --netG unet_256 \
    --direction AtoB --dataset_mode aligned \
    --gan_mode vanilla --norm batch --pool_size 0
