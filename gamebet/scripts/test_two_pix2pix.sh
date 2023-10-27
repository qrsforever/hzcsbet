#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

GPUID=-1

cd ${TOP_DIR}
python3 ./test_two_pix2pix.py --dataroot ./datasets/soccer_seg_detection/single \
    --gpu_ids ${GPUID} --output_nc 1 --load_size 256 \
    --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch
