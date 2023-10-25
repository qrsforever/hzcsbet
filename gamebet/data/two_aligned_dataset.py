#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file two_aligned_dataset.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-10-24 19:39

from data.aligned_dataset import AlignedDataset
from data.base_dataset import BaseDataset

class TwoAlignedDataset(BaseDataset):
    def initialize(self, opt):
        assert opt.isTrain is True       
        opt1 = opt
        opt1.phase = opt.phase1
        opt1.dataset_model = 'aligned'        
        self.dataset1 = AlignedDataset()
        self.dataset1.initialize(opt1)

        opt2 = opt
        opt2.phase = opt.phase2
        opt2.dataset_model = 'aligned'
        self.dataset2 = AlignedDataset()
        self.dataset2.initialize(opt2)

    def __getitem__(self, index):        
        return {
            'dataset1_input': self.dataset1[index],
            'dataset2_input': self.dataset2[index]
        }
       
    def __len__(self):
        assert len(self.dataset1) == len(self.dataset2)
        return len(self.dataset1)
    
    def name(self):
        return 'TwoAlignedDataset'
