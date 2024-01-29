#!/usr/bin/env python3

from detector import DetectExecutor
from tracking import TrackExecutor
from teamfier import TeamfierExecutor
from sinking import SinkingExecutor
from pix2pix import Pix2PixSegExecutor, Pix2PixDetExecutor

from aux.executor import ExecutorBase
from aux.message import SharedResult

import aux.constant as C

import os
import logging
import multiprocessing as mp
import numpy as np
import torch

from multiprocessing.shared_memory import SharedMemory
from logging.handlers import QueueListener


def logger_process(queue):
    console = logging.StreamHandler()
    console.setFormatter(C.LOGGER_FORMAT)
    logger = logging.getLogger('app')
    logger.addHandler(console)
    logger.setLevel(C.LOGGER_LEVEL)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


class VideoExecutor(ExecutorBase):
    _name = "Master"

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self, frame, msg, cache):
        if msg.token > 0:
            self.logger.debug(msg)
        success, image = cache['videocap'].read()
        # only debug
        if cache['current_count'] == 50:
            return None
        if not success:
            return None
        frame[:] = image
        cache['current_count'] += 1
        return msg.reset(cache['current_count'])

    def pre_loop(self, cache):
        import cv2
        if self.video_path is None or not os.path.exists(self.video_path):
            self.send(None)
            self.logger.error(f'video path: {self.video_path} is not valid!!!')
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.send(None)
            self.logger.error(f'{self.video_path} is not valid!!!')
            return
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        shape = (frame_height, frame_width, 3)
        size = np.prod(shape) * np.dtype('uint8').itemsize
        for i in range(C.SHM_CACHE_COUNT):
            name = 'cache-%d' % i
            if not os.path.exists(f'/dev/shm/{name}'):
                shared_mem = SharedMemory(name=name, create=True, size=size)
                shared_mem.close()
            self.send(SharedResult(name, frame_width, frame_height))

        cache['frame_count'] = frame_count
        cache['frame_rate'] = frame_rate
        cache['current_count'] = 0
        cache['videocap'] = cap
        self.logger.info(f'pre cache: {cache}')

    def post_loop(self, cache):
        self.logger.info(f'post cache: {cache}')
        for i in range(C.SHM_CACHE_COUNT):
            name = 'cache-%d' % i
            shared_mem = SharedMemory(name=name, create=False)
            shared_mem.close()
            shared_mem.unlink()
        if 'videocap' in cache:
            cache['videocap'].release()
        self.logger.info(f'{self.name} post finished')


def main():

    logger_queue = mp.Queue()
    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(C.LOGGER_FORMAT)
    queue_listener = QueueListener(logger_queue, logger_handler)
    queue_listener.start()

    use_gpu = torch.cuda.is_available()

    print("Main Starting...")
    sinking = SinkingExecutor()
    teamfier = TeamfierExecutor()
    tracking = TrackExecutor()
    detector = DetectExecutor(C.YOLO_DETECT_WEIGHTS_PATH, conf=0.2)
    pix2seg  = Pix2PixSegExecutor(C.PIX2PIX_SEG_WEIGHTS_PATH, use_gpu=use_gpu)
    mainproc = VideoExecutor(video_path=C.VIDEO_PATH)

    mainproc.linkto(pix2seg).linkto(detector).linkto(sinking).linkto(mainproc, False)
    # mainproc.linkto(detector).linkto(tracking).linkto(teamfier).linkto(sinking).linkto(mainproc, False)

    mainproc.start(logger_queue)
    mainproc.join()

    print("Main Finished!")


if __name__ == '__main__':
    main()
