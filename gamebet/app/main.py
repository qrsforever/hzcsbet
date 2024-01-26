#!/usr/bin/env python3

from detector import DetectExecutor
from tracking import TrackExecutor
from teamfier import TeamfierExecutor

from aux.executor import ExecutorBase
from aux.message import SharedResult
from aux.constant import LOGGER_LEVEL, LOGGER_FORMAT

import logging
import multiprocessing as mp

from logging.handlers import QueueHandler


def logger_process(queue):
    console = logging.StreamHandler()
    console.setFormatter(LOGGER_FORMAT)
    logger = logging.getLogger('app')
    logger.addHandler(console)
    logger.setLevel(LOGGER_LEVEL)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)
        import time
        time.sleep(0.3)


class MainExecutor(ExecutorBase):
    def run(self, msg, logger):
        if msg == -1:
            return None
        return SharedResult('shm-%d' % msg)


def main():
    # logger_queue = mp.Queue()
    logger = logging.getLogger('app')
    # logger.addHandler(QueueHandler(logger_queue))
    # logger.setLevel(LOGGER_LEVEL)
    # logger_proc = mp.Process(target=logger_process, args=(logger_queue,))
    # logger_proc.start()
    logger.info('Main process start.')

    teamfier = TeamfierExecutor('teamfier', None)
    tracking = TrackExecutor('tracking', teamfier)
    detector = DetectExecutor('detector', tracking)
    mainproc = MainExecutor('mainproc', detector)
    logger_queue = None
    mainproc.start(logger_queue)
    for i in range(1):
        logger.info('send %d' % i)
        mainproc.send(i)
    mainproc.send(-1)
    mainproc.join()
    logger.info('Main process done.')
    # logger_queue.put(None)


if __name__ == '__main__':
    main()
