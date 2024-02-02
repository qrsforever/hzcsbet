#!/usr/bin/env python3

from sources import (
    VideoExecutor,
    ImageExecutor,
)

from detector import DetectExecutor
from tracking import TrackExecutor
from teamfier import TeamfierExecutor
from sinking import (
    BlendSinkExecutor,
    GridSinkExecutor
)
from pix2pix import (
    Pix2PixSegExecutor,
    Pix2PixDetExecutor,
    Pix2PixDetSiameseExecutor,
    Pix2PixSegDetSiameseExecutor,
)
from birdeyeview import (
    BEVDeepExecutor,
    BEVHogExecutor,
)

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C

import os
import logging
import multiprocessing as mp
import torch
import argparse

from logging.handlers import QueueListener


def main():

    use_gpu = torch.cuda.is_available()
    torch.multiprocessing.set_start_method('forkserver', force=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type=str, required=True, help='video file path or image dir path')
    parser.add_argument('--start', type=int, default=-1, help='start at x seconds if video, count if images')
    parser.add_argument('--stop', type=int, default=-1, help='stop at x secodns if video, count if images')
    args = parser.parse_args()

    logger_queue = mp.Queue()
    logger_queue = mp.Queue()
    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(C.LOGGER_FORMAT)
    queue_listener = QueueListener(logger_queue, logger_handler)
    queue_listener.start()

    print("Main Starting...")
    # sinking = BlendSinkExecutor(video_output_path=f'{C.APP_OUTPUT_PATH}/blend_out.mp4')
    sinking = GridSinkExecutor(video_output_path=f'{C.APP_OUTPUT_PATH}/blend_grid_out.mp4')
    teamfier = TeamfierExecutor()
    tracking = TrackExecutor()
    detector = DetectExecutor(conf=0.2)
    pix2seg = Pix2PixSegExecutor(use_gpu=use_gpu)
    pix2det = Pix2PixDetExecutor(use_gpu=use_gpu)
    # pix2detsme = Pix2PixDetSiameseExecutor(use_gpu=use_gpu)
    # pix2pix = Pix2PixSegDetSiameseExecutor(use_gpu=use_gpu, blend_seg=True, blend_det=False)
    birdeye = BEVHogExecutor()
    if os.path.isfile(args.source):
        source = VideoExecutor(args.source, args.start, args.stop)
    else:
        source = ImageExecutor(args.source, args.start, args.stop)

    source.linkto(birdeye).linkto(sinking).linkto(source, False)
    # source.linkto(pix2seg).linkto(pix2det).linkto(birdeye).linkto(sinking).linkto(source, False)
    # source.linkto(pix2pix).linkto(detector).linkto(tracking).linkto(teamfier).linkto(sinking).linkto(source, False)
    # source.linkto(pix2seg).linkto(pix2detsme).linkto(sinking).linkto(source, False)
    # source.linkto(pix2pix).linkto(detector).linkto(tracking).linkto(teamfier).linkto(birdeye).linkto(sinking).linkto(source, False)

    source.start(logger_queue)
    source.join()

    print("Main Finished!")


if __name__ == '__main__':
    main()
