#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

PHASE=${1:-seg}
GPUID=-1

if [[ x$PHASE == xseg ]]
then
    PHASE_DATASET=train_phase_1
else
    PHASE_DATASET=train_phase_2
fi

python3 ${TOP_DIR}/train_pix2pix.py --dataroot ${TOP_DIR}/datasets/soccer_seg_detection \
    --gpu_ids ${GPUID} --phase ${PHASE_DATASET} --display_id -1 --print_freq 10 \
    --name soccer_${PHASE}_pix2pix --model pix2pix --netG unet_256 \
    --direction AtoB --dataset_mode aligned --output_nc 1 --n_epochs 200 \
    --gan_mode vanilla --norm batch --pool_size 0 --save_epoch_fre 100
