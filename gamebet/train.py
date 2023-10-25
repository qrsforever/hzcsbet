#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file train_two_pix2pix.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-10-24 21:18

from options.train_options import TrainOptions
from data.two_aligned_dataset import TwoAlignedDataset


opt = TrainOptions().parse()
