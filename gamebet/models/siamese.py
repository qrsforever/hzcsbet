#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file siamese.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-14 16:34


import torch.nn.functional as F
import torch.nn as nn
import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin:float):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        """
        similar:1, dissimilar:0
        """
        dist = F.pairwise_distance(x1, x2, keepdim=True)
        loss = torch.mean(
            label * torch.pow(dist, 2) + \
                (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss


class SiameseNetwork(nn.Module):
    def __init__(self):
        """
        inputs: (B, 1, 180, 320)

        filter_feature_size: W / ((K - 2 * Pad) * S), H / ((K - 2 * Pad) * S)
        """
        super(SiameseNetwork, self).__init__()
        self.extract_features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1, inplace=True),
            # (B, 4, 90, 160)
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            # (B, 8, 45, 80)
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # (B, 16, 23, 40)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # (B, 32, 12, 20)
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # (B, 16, 6, 10)
            nn.Dropout(p=0.3),
        )
        self.fc = nn.Sequential(
            # nn.Linear(960, 16), # 960: 6 * 10 * 16
            nn.Linear(960, 480), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(480, 240), nn.ReLU(inplace=True), # nn.Dropout(p=0.5),
            nn.Linear(240, 16)
        )

    def _forward_once(self, x):
        x = self.extract_features(x)
        x = self.fc(x.view(x.size(0), -1))
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        x1 = self._forward_once(x1)
        x2 = self._forward_once(x2)
        return x1, x2
