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
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data.camera_pair_dataset import CameraPair, CameraPairDataset 
from models.siamese import ContrastiveLoss, SiameseNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True, type=str, help='a .mat file')
parser.add_argument('--cuda-id', required=False, type=int, default=0, help='CUDA ID 0, 1, 2, 3')
parser.add_argument('--lr', required=True, type=float, default=0.01, help='learning rate')
parser.add_argument('--num-epoch', required=True, type=int, help='epoch number')
parser.add_argument('--batch-size', required=True, type=int)

args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise

dataset_sample = sio.loadmat(args.data_path)

## Dataset

data_transform = transforms.Compose([
    transforms.Resize((180, 320)), # H, W
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_sample['image_mean'], std=dataset_sample['image_std'])
])

data = CameraPair(anchor_images=dataset_sample['pivot_images'], positive_images=dataset_sample['positive_images'])
dataset = CameraPairDataset(data, transform=data_transform)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

## Loss

learning_rate = args.lr
criterion = ContrastiveLoss(margin=1.0)

## Optimizer

net = SiameseNetwork()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()),
    lr=learning_rate, weight_decay=0.000001
)

## Train

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.cuda_id))
    net = net.to(device)
    criterion = criterion.cuda(device)
    cudnn.benchmark = True

for epoch in range(args.num_epoch):
    net.train()
    losses = []
    for x1, x2, label in train_loader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        feat1, feat2 = net(x1, x2)
        loss = criterion(feat1, feat2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print('[%d] loss = %.5f' % (epoch, sum(losses) / max(1, len(losses))))
