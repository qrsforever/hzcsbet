#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file test_siamese.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-17 14:54


import os
import scipy.io as sio
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torch.nn.functional as F

from data.camera_pair_dataset import CameraPair, CameraPairDataset
from models.siamese import SiameseNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True, type=str, help='a .mat file')
parser.add_argument('--cuda-id', required=False, type=int, default=0, help='CUDA ID 0, 1, 2, 3')
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--num-workers', required=False, type=int, default=4, help='job worker number')
parser.add_argument('--weights-path', default='', type=str, help='path of checkpoint weights')

args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise

dataset_sample = sio.loadmat(args.data_path)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.cuda_id))
    cudnn.benchmark = True

model_state_dict = None
if os.path.exists(args.weights_path):
    ckpts = torch.load(args.weights_path, map_location=device)
    model_state_dict = ckpts['model']
else:
    raise

## Dataset

data_transform = transforms.Compose([
    transforms.Resize((180, 320)), # H, W
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.01848], std=[0.11606])
])

data = CameraPair(anchor_images=dataset_sample['pivot_images'], positive_images=dataset_sample['positive_images'])
dataset = CameraPairDataset(data, transform=data_transform)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

siamese = SiameseNetwork().to(device)
if model_state_dict is not None:
    siamese.load_state_dict(model_state_dict)

## Test
siamese.eval()
with torch.no_grad():
    pos_dists, neg_dists = [], []
    for bx1, bx2, labels in test_loader:
        bx1, bx2, labels = bx1.to(device), bx2.to(device), labels.to(device)
        feat1, feat2 = siamese(bx1, bx2)
        dist = F.pairwise_distance(feat1.detach(), feat2.detach(), keepdim=True)
        for a, b in list(zip(dist.cpu().detach().numpy(), labels.cpu().detach().numpy())):
            print(float(a), float(b))
        for l, d in zip(labels.detach().squeeze(), dist.squeeze()):
            pos_dists.append(d) if l == 1 else neg_dists.append(d)

    dist_pos, dist_neg = torch.mean(torch.tensor(pos_dists)), torch.mean(torch.tensor(neg_dists))
    print('pos_d[%.6f] neg_d[%.6f] ratio[%.6f]' % (dist_pos, dist_neg, dist_neg / (dist_pos + 0.000001)))
