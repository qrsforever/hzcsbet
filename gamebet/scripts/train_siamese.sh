#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

python3 ${TOP_DIR}/train_siamese.py --data-path ${TOP_DIR}/data/dataset_sample.mat \
    --lr 0.0001 --num-epoch 100 --batch-size 16
