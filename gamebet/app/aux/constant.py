#!/usr/bin/env python3

import logging
import os
from dataclasses import dataclass

## logger

LOGGER_NAME_TO_LEVEL = {
    'debug': logging.DEBUG, 'd': logging.DEBUG, 'D': logging.DEBUG, '10': logging.DEBUG,
    'info': logging.INFO, 'i': logging.INFO, 'I': logging.INFO, '20': logging.INFO,
    'warn': logging.WARN, 'w': logging.WARN, 'W': logging.WARN, '30': logging.WARN,
    'error': logging.ERROR, 'e': logging.ERROR, 'E': logging.ERROR, '40': logging.ERROR,
}

LOGGER_LEVEL = LOGGER_NAME_TO_LEVEL[os.environ.get('LOG', 'info')]
LOGGER_FORMAT = logging.Formatter("%(asctime)s - %(process)s - %(levelname)s: %(message)s")

## app & shared memory

SHM_CACHE_COUNT = 48

APP_OUTPUT_PATH = os.environ.get('APP_OUTPUT_PATH', 'output')

## Video

VIDEO_INPUT_PATH = os.environ.get('VIDEO_INPUT_PATH', '')
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 30

## Weights

CKPTS_ROOT_PATH = os.environ.get("CHECKPOINTS_ROOT_PATH")
YOLO_DETECT_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/soccer_yolov8x.pt'
PIX2PIX_SEG_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/seg_net_G.pth'
PIX2PIX_DET_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/det_net_G.pth'
SIAMESE_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/siamese.pth'


## tracking

@dataclass(frozen=True)
class ByteTrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

IND_TO_CLS = {
    0: "ball",
    1: "player",
    2: "referee",
    3: "goalkeeper",
}

CLS_TO_IND = {
    "ball": 0,
    "player": 1,
    "referee": 2,
    "goalkeeper": 3,
}

DETECTION_COLORS = {
    "ball": (0, 200, 200),
    "player": (255, 0, 0),
    "goalkeeper": (255, 0, 255),
    "referee": (0, 0, 255),
}

## color cluster

COLOR_CLUSTER_BOUNDARIES = [
    ([43, 31, 4], [128, 0, 0], [250, 88, 50]),           # blue
    ([0, 100, 0], [0, 128, 0], [50, 255, 50]),           # green
    ([17, 15, 100], [0, 0, 255], [50, 56, 200]),         # red
    ([192, 192, 0], [192, 192, 0], [255, 255, 128]),     # cyan
    ([192, 0, 192], [192, 0, 192], [255, 128, 255]),     # magenta
    ([0, 192, 192], [0, 192, 192], [128, 255, 255]),     # yellow
    ([0, 0, 0], [0, 0, 0], [50, 50, 50]),                # black
    ([187, 169, 112], [255, 255, 255], [255, 255, 255]), # white
]

## pix2pix seg

@dataclass
class Pix2PixArgs:
    load_size: int = 256

# ----------------- Options ---------------
#              aspect_ratio: 1.0                           
#                batch_size: 1                             
#           checkpoints_dir: ./checkpoints                 
#                 crop_size: 256                           
#                  dataroot: ./datasets/soccer_seg_detection/single	[default: None]
#              dataset_mode: single                        
#                 direction: AtoB                          
#           display_winsize: 256                           
#                     epoch: latest                        
#                      eval: False                         
#                   gpu_ids: -1                            	[default: 0]
#                 init_gain: 0.02                          
#                 init_type: normal                        
#                  input_nc: 3                             
#                   isTrain: False                         	[default: None]
#                 load_iter: 0                             	[default: 0]
#                 load_size: 256                           
#          max_dataset_size: inf                           
#                     model: test                          
#              model_suffix:                               
#                n_layers_D: 3                             
#                      name: experiment_name               
#                       ndf: 64                            
#                      netD: basic                         
#                      netG: unet_256                      	[default: resnet_9blocks]
#                       ngf: 64                            
#                no_dropout: False                         
#                   no_flip: False                         
#                      norm: batch                         	[default: instance]
#                  num_test: 50                            
#               num_threads: 4                             
#                 output_nc: 1                             	[default: 3]
#                     phase: test                          
#                preprocess: resize_and_crop               
#               results_dir: ./results/                    
#            serial_batches: False                         
#                    suffix:                               
#                 use_wandb: False                         
#                   verbose: False                         
#        wandb_project_name: CycleGAN-and-pix2pix          
# ----------------- End -------------------
