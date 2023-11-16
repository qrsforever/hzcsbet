#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

[ ! -d mkdir ${TOP_DIR}/checkpoints ] && mkdir ${TOP_DIR}/checkpoints

python3 ${TOP_DIR}/train_siamese.py --data-path ${TOP_DIR}/data/dataset_sample.mat \
    --lr 0.1 --num-epoch 300 --batch-size 512 --resume-from ${TOP_DIR}/checkpoints/siamese.pth
