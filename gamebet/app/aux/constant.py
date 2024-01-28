#!/usr/bin/env python3

import logging
import os
from dataclasses import dataclass


LOGGER_NAME_TO_LEVEL = {
    'debug': logging.DEBUG, 'd': logging.DEBUG, 'D': logging.DEBUG, '10': logging.DEBUG,
    'info': logging.INFO, 'i': logging.INFO, 'I': logging.INFO, '20': logging.INFO,
    'warn': logging.WARN, 'w': logging.WARN, 'W': logging.WARN, '30': logging.WARN,
    'error': logging.ERROR, 'e': logging.ERROR, 'E': logging.ERROR, '40': logging.ERROR,
}

LOGGER_LEVEL = LOGGER_NAME_TO_LEVEL[os.environ.get('LOG', 'info')]
LOGGER_FORMAT = logging.Formatter("%(asctime)s - %(process)s - %(levelname)s: %(message)s")

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 30

SHM_CACHE_COUNT = 48


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

CLUSTER_BOUNDARIES = [
    ([43, 31, 4], [128, 0, 0], [250, 88, 50]),           # blue
    ([0, 100, 0], [0, 128, 0], [50, 255, 50]),           # green
    ([17, 15, 100], [0, 0, 255], [50, 56, 200]),         # red
    ([192, 192, 0], [192, 192, 0], [255, 255, 128]),     # cyan
    ([192, 0, 192], [192, 0, 192], [255, 128, 255]),     # magenta
    ([0, 192, 192], [0, 192, 192], [128, 255, 255]),     # yellow
    ([0, 0, 0], [0, 0, 0], [50, 50, 50]),                # black
    ([187, 169, 112], [255, 255, 255], [255, 255, 255]), # white
]
