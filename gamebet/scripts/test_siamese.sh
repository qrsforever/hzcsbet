#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TOP_DIR=$(cd ${CUR_DIR}/..; pwd)

python3 ${TOP_DIR}/test_siamese.py --data-path ${TOP_DIR}/data/test_sample.mat \
    --batch-size 16 --num-workers $(nproc) \
    --weights-path ${TOP_DIR}/checkpoints/siamese_10000.pth
