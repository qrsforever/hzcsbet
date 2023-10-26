#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file train_two_pix2pix.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-10-24 21:18

from options.train_options import TrainOptions
from data import create_dataset


if __name__ == "__main__":
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    print('dataset size = %d' % len(dataset))

    for i, data in enumerate(dataset):
        print(i, data)
        break


