#!/usr/bin/env python3

import os
TOP_DIR='/data/k12ai/codes/hzcsbet'
os.environ['PYTHONPATH'] = TOP_DIR

import scipy.io as sio
from util.synthetic import generate_siamese_dataset

N = 9999

camera_parameter_file = f'{TOP_DIR}/gamebet/data/worldcup_dataset_camera_parameter.mat'
soccer_field_template_file = f'{TOP_DIR}/gamebet/data/worldcup2014.mat'
dataset_sample_file = f'{TOP_DIR}/gamebet/data/dataset_sample_{N}.mat'

generate_siamese_dataset(camera_parameter_file, soccer_field_template_file, dataset_sample_file, image_num=N)
dataset_sample = sio.loadmat(dataset_sample_file)
print(dataset_sample['image_mean'], dataset_sample['image_std'])
