#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file camera_pair_dataset.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-13 20:10


import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from collections import namedtuple

CameraPair = namedtuple('CameraPair', ['anchor_images', 'positive_images'])

class CameraPairDataset(Dataset):

    def __init__(self, data:CameraPair, transform=None):
        self.data = data
        self.size = len(self.data.anchor_images)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((180, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0188], std=[0.128])
            ])
        else:
            self.transform = transform

        #
        # | 0, 1, 2, ..., N, 0, 1, 2, ..., N | --> x1
        # | 0, 1, 2, ..., N, roll(shift=1)   | --> x2
        # | 1, 1, 1, ..., 1, 0, 0, 0, ..., 0 | --> label
        #
        rng = np.random.default_rng(seed=123456)
        indices = rng.permutation(self.size)
        x1_inputs = np.concatenate((indices, indices), axis=0).reshape(-1, 1)
        x2_inputs = np.concatenate((indices, np.roll(indices, shift=1)), axis=0).reshape(-1, 1)
        self.indices = rng.permutation(np.concatenate((x1_inputs, x2_inputs), axis=1))

    def __len__(self):
        return 2 * self.size

    def __getitem__(self, index):
        ix1, ix2 = self.indices[index]
        x1 = self.data.anchor_images[ix1]
        x2 = self.data.anchor_images[ix2] if ix1 != ix2 else self.data.positive_images[ix2]
        return (self.transform(Image.fromarray(x1)), \
                self.transform(Image.fromarray(x2)), \
                torch.from_numpy(np.array([int(ix1 == ix2)], dtype=np.float32)))


class FieldEdgeImages(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.size = len(data)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((180, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0188], std=[0.128])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.data[index]))
