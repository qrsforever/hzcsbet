#!/usr/bin/env python3

from .constant import LOGGER_LEVEL, LOGGER_FORMAT

import abc
import logging
import multiprocessing as mp

from logging.handlers import QueueHandler


class ExecutorBase(abc.ABC):
    def __init__(self, name: str, next = None):
        self._name = name
        if next is not None:
            self._out_queue = next._in_queue
            self._next = next
        else:
            self._next = None
            self._out_queue = None
        self._in_queue = mp.Queue()
        self._proc = None

    def send(self, msg):
        return self._in_queue.put(msg)

    @abc.abstractmethod
    def run(self, msg, logger=None):
        return msg

    def _task_run(self, log_queue: mp.Queue, in_queue: mp.Queue, out_queue: mp.Queue) -> None:
        self._name = mp.current_process().name
        if log_queue is None:
            console = logging.StreamHandler()
            console.setFormatter(LOGGER_FORMAT)
            logger = mp.get_logger()
            logger.addHandler(console)
            logger.setLevel(LOGGER_LEVEL)
        else:
            logger = logging.getLogger('app')
            logger.addHandler(QueueHandler(log_queue))
            logger.setLevel(LOGGER_LEVEL)
        logger.info(f'{self._name} task is running!')
        while True:
            msg = in_queue.get()
            if isinstance(msg, str):
                if msg == 'q':
                    break
            if msg is None:
                if out_queue is not None:
                    out_queue.put(None)
                break
            if out_queue is not None:
                out_queue.put(self.run(msg, logger))

    def start(self, log_queue=None):
        if self._next is not None:
            self._next.start(log_queue)
        if self._proc is None:
            self._proc = mp.Process(
                name=self._name,
                target=self._task_run,
                args=(log_queue, self._in_queue, self._out_queue))
            self._proc.start()

    def join(self):
        if self._next is not None:
            self._next.join()
        if self._proc is not None:
            self._proc.join()
