#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file train_siamese.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-14 21:44


import os
import scipy.io as sio
import argparse
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torch.nn.functional as F

from data.camera_pair_dataset import CameraPair, CameraPairDataset
from models.siamese import ContrastiveLoss, SiameseNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True, type=str, help='a .mat file')
parser.add_argument('--cuda-id', required=False, type=int, default=0, help='CUDA ID 0, 1, 2, 3')
parser.add_argument('--lr', required=True, type=float, default=0.01, help='learning rate')
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--num-epoch', required=True, type=int, help='epoch number')
parser.add_argument('--num-workers', required=False, type=int, default=4, help='job worker number')

parser.add_argument('--resume-from', default='', type=str, help='path to the save checkpoint')

args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise

dataset_sample = sio.loadmat(args.data_path)

model_save_path = args.resume_from

epoch_beg = 0
epoch_num = args.num_epoch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.cuda_id))
    cudnn.benchmark = True

model_state_dict, optim_state_dict = None, None
if os.path.exists(args.resume_from):
    ckpts = torch.load(args.resume_from, map_location=device)
    model_state_dict, optim_state_dict, epoch_beg = ckpts['model'], ckpts['optimizer'], ckpts['epoch']

print('start from: ', epoch_beg, ' mean: ', dataset_sample['image_mean'], ' std: ', dataset_sample['image_std'])

## Dataset

data_transform = transforms.Compose([
    transforms.Resize((180, 320)), # H, W
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.01848], std=[0.11606])
])

data = CameraPair(anchor_images=dataset_sample['pivot_images'], positive_images=dataset_sample['positive_images'])
dataset = CameraPairDataset(data, transform=data_transform)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

## Loss

learning_rate = args.lr
criterion = ContrastiveLoss(margin=1.0).to(device)

## Optimizer

siamese = SiameseNetwork().to(device)
if model_state_dict is not None:
    siamese.load_state_dict(model_state_dict)

# optimizer = optim.SGD(siamese.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(siamese.parameters(), lr=learning_rate)
if optim_state_dict is not None:
    optimizer.load_state_dict(optim_state_dict)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 45, 60], gamma=0.1)

## Train

for epoch in range(epoch_beg, epoch_num + 1):
    siamese.train()
    losses, pos_dists, neg_dists = [], [], []
    for bx1, bx2, labels in train_loader:
        bx1, bx2, labels = bx1.to(device), bx2.to(device), labels.to(device)
        feat1, feat2 = siamese(bx1, bx2)
        loss = criterion(feat1, feat2, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        dist = F.pairwise_distance(feat1.detach(), feat2.detach(), keepdim=True)
        for l, d in zip(labels.detach().squeeze(), dist.squeeze()):
            pos_dists.append(d) if l == 1 else neg_dists.append(d)

    scheduler.step()

    dist_pos, dist_neg = torch.mean(torch.tensor(pos_dists)), torch.mean(torch.tensor(neg_dists))
    print('[%d] lr=[%.6f] loss[%.6f] pos_d[%.6f] neg_d[%.6f] ratio[%.6f]' % (
        epoch,
        optimizer.param_groups[0]['lr'],
        torch.mean(torch.tensor(losses)), dist_pos, dist_neg,
        dist_neg / (dist_pos + 0.000001)
        ))

    if (epoch + 1) % 60 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model': siamese.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_save_path)

torch.save({
    'epoch': epoch + 1,
    'model': siamese.state_dict(),
    'optimizer': optimizer.state_dict()
}, model_save_path)
