#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

[ ! -d ${TOP_DIR}/checkpoints ] && mkdir ${TOP_DIR}/checkpoints

python3 ${TOP_DIR}/train_siamese.py --data-path ${TOP_DIR}/data/dataset_sample_10000.mat \
    --lr 0.1 --num-epoch 75 --batch-size 512 --num-workers $(nproc) \
    --resume-from ${TOP_DIR}/checkpoints/siamese_10000.pth
