#!/usr/bin/env python3

from .constant import LOGGER_LEVEL, LOGGER_FORMAT

import abc
import logging
import multiprocessing as mp
import numpy as np

from multiprocessing.shared_memory import SharedMemory
from logging.handlers import QueueHandler
from contextlib import contextmanager
from multiprocessing.resource_tracker import unregister


@contextmanager
def shared_memory_to_numpy(msg):
    try:
        shape = (msg.frame_height, msg.frame_width, 3)
        shared_mem = SharedMemory(name=msg.shm_name, create=False)
        np_array = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
        yield np_array
    finally:
        # TODO: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
        # https://forums.raspberrypi.com/viewtopic.php?t=340441
        # unregister(shared_mem._name, 'shared_memory') # type: ignore
        shared_mem.close()


class ExecutorBase(abc.ABC):
    _name: str = '__base__'

    def __init__(self):
        self._proc = None
        self._iq, self._oq, self._next = None, None, None

    def linkto(self, to, next=True):
        to._iq = self._oq = mp.Queue()
        if next:
            self._next = to
        return to

    def send(self, msg):
        if self._iq is not None:
            return self._iq.put(msg)

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def run(self, frame, msg, cache):
        return msg

    def pre_loop(self, cache):
        pass

    def post_loop(self, cache):
        pass

    # TODO: https://stackoverflow.com/questions/62830911/typeerror-cannot-pickle-weakref-object
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_proc'] = None
        return state

    def _task_run(self, log_queue: mp.Queue, in_queue: mp.Queue, out_queue: mp.Queue) -> None:
        # self._name = mp.current_process().name
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
        self.logger = logger
        self.logger.info(f'{self.name} task is running!')
        cache = {}
        import time
        import torch
        torch.backends.cudnn.benchmark = True
        self.pre_loop(cache)
        # TODO shm resource_tracker deletation issue
        shmid_cache = set()
        while True:
            msg = in_queue.get()
            if msg is None:
                if out_queue is not None:
                    out_queue.put(None)
                break
            self.logger.debug(f'{self._name} get a message cache: {msg._shmid}.')
            shmid_cache.add(msg._shmid)
            if out_queue is not None:
                with shared_memory_to_numpy(msg) as frame:
                    try:
                        t0 = time.time()
                        out = self.run(frame, msg, cache)
                        self.logger.debug(f'{self._name} timeit: {time.time() - t0}')
                        out_queue.put(out)
                    except Exception as err:
                        self.logger.error(f'{err}')
                        out_queue.put(None)
                        break
        self.logger.warn(f'{self._name} quit loop!')
        for shmid in shmid_cache:
            try:
                # https://forums.raspberrypi.com/viewtopic.php?t=340441
                unregister('/' + shmid, 'shared_memory')
            except:
                pass
        self.post_loop(cache)

    def start(self, log_queue=None):
        if self._next is not None:
            self._next.start(log_queue)
        if self._proc is None:
            self._proc = mp.Process(
                name=self._name,
                target=self._task_run,
                args=(log_queue, self._iq, self._oq))
            self._proc.start()

    def join(self):
        if self._next is not None:
            self._next.join()
        if self._proc is not None:
            self._proc.join()
