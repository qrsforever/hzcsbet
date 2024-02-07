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

SHM_CACHE_COUNT = 24
SHM_FRAME_COUNT = 2

APP_OUTPUT_PATH = os.environ.get('APP_OUTPUT_PATH', 'output')

## Video

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 25

## Weights

CKPTS_ROOT_PATH = os.environ.get("CHECKPOINTS_ROOT_PATH")
YOLO_DETECT_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/soccer_yolov8x.pt'
PIX2PIX_SEG_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/seg_net_G.pth'
PIX2PIX_DET_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/det_net_G.pth'
SIAMESE_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/siamese.pth'
ROUBUST_SFR_WEIGHTS_PATH = f'{CKPTS_ROOT_PATH}/robust_sfr.pth'
FEATURES_CAMERAS_PARAMS_PATH = f'{CKPTS_ROOT_PATH}/features_cameras_params.mat'
FEATURES_CAMERAS_HOG_PARAMS_PATH = f'{CKPTS_ROOT_PATH}/features_cameras_hog_params.mat'

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

## camera

WORLDCUP2014_TEMPL_PATH = f'{CKPTS_ROOT_PATH}/worldcup2014.mat'

## BirdsEye

# yard2meter = 0.9144
# template_h, template_w = int(74 * yard2meter) + 2, int(115 * yard2meter) + 2

TEMPLATE_W, TEMPLATE_H = 115, 74
RENDER_W, RENDER_H = 1050, 680

BG_BLACK_FIELD_PATH = f'{CKPTS_ROOT_PATH}/field_black.jpg'
BG_GREEN_FIELD_PATH = f'{CKPTS_ROOT_PATH}/field_green.jpg'
