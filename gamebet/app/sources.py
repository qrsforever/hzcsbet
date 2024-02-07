#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np
import os
import cv2

from multiprocessing.shared_memory import SharedMemory


class SourceExecutor(ExecutorBase):
    _name = "Source"

    def __init__(self, source_path: str):
        super().__init__()
        self._source_path = source_path
        self._frame_scale: float = -1
        self._frame_width = C.FRAME_WIDTH
        self._frame_height = C.FRAME_HEIGHT

    def run(self, shared_frames: tuple[np.ndarray, ...], msg, cache):
        if msg.token > 0:
            self.logger.debug(msg)
        if cache['current_count'] >= cache['frame_stop']:
            self.logger.debug(f'frame {msg.token}/{cache["frame_stop"]} finished.')
            return None
        success, image = cache['reader'].read()
        if not success:
            return None
        if self._frame_scale != 1.0:
            image = cv2.resize(image, (self._frame_width, self._frame_height), interpolation=cv2.INTER_AREA)

        for i in range(C.SHM_FRAME_COUNT):
            frame = shared_frames[i]
            frame[:] = image
        cache['current_count'] += 1
        self.logger.debug(f'frame {cache["current_count"]}/{cache["frame_stop"]} finished.')
        return msg.reset(cache['current_count'])

    def pre_loop(self, cache):
        shape = (self._frame_height, self._frame_width, 3)
        size = C.SHM_FRAME_COUNT * np.prod(shape) * np.dtype('uint8').itemsize
        for i in range(C.SHM_CACHE_COUNT):
            name = 'cache-%d' % i
            if not os.path.exists(f'/dev/shm/{name}'):
                shared_mem = SharedMemory(name=name, create=True, size=size)  # pyright: ignore[reportArgumentType]
                shared_mem.close()
            self.send(SharedResult(name, size, shape))

    def post_loop(self, cache):
        self.logger.info(f'post cache: {cache}')
        for i in range(C.SHM_CACHE_COUNT):
            name = 'cache-%d' % i
            shared_mem = SharedMemory(name=name, create=False)
            shared_mem.close()
            shared_mem.unlink()


class VideoExecutor(SourceExecutor):
    _name = "VideoSource"

    def __init__(self, source_path: str, from_time_secs=-1, to_time_secs=-1):
        super().__init__(source_path)
        self._from_time_secs = 0 if from_time_secs < 0 else from_time_secs
        self._to_time_secs = to_time_secs

    def pre_loop(self, cache):# {{{
        if self._source_path is None or not os.path.exists(self._source_path):
            self.send(None)
            self.logger.error(f'video path: {self._source_path} is not valid!!!')
            return
        cap = cv2.VideoCapture(self._source_path)
        if not cap.isOpened():
            self.send(None)
            self.logger.error(f'{self._source_path} is not valid!!!')
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_from = self._from_time_secs * frame_rate
        frame_stop = frame_count if self._to_time_secs < 0 else self._to_time_secs * frame_rate
        if frame_from >= frame_count or frame_stop < frame_from:
            self.send(None)
            self.logger.error(f'[{frame_from} exceed the max frame count[{frame_count}]!!!')
            return
        if frame_from > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_from)
        if frame_width != self._frame_width:
            self._frame_scale = round(frame_width / self._frame_width, 3)
            self.logger.warn(f'({self._frame_width}, {self._frame_height}) vs ({frame_width}, {frame_height})')

        cache['frame_count'] = frame_count
        cache['frame_rate'] = int(frame_rate)
        cache['frame_from'] = int(frame_from)
        cache['frame_stop'] = int(frame_stop)
        cache['current_count'] = int(frame_from)
        cache['reader'] = cap
        super().pre_loop(cache)
        self.logger.info(f'pre cache: {cache}')# }}}

    def post_loop(self, cache):# {{{
        if 'cap' in cache:
            cache['cap'].release()
        super().post_loop(cache)
        self.logger.info(f'{self.name} post finished')# }}}


class ImageExecutor(SourceExecutor):
    _name = "ImageSource"

    def __init__(self, source_path: str, from_index=-1, to_index=-1):
        super().__init__(source_path)
        self._from_index = 1 if from_index <= 0 else from_index
        self._to_index = to_index

    # Only Debug
    # def run(self, shared_frames: tuple[np.ndarray, ...], msg, cache):
    #     if msg.token > 0:
    #         self.logger.debug(msg)
    #     if cache['current_count'] >= cache['frame_stop']:
    #         self.logger.info(cache)
    #         return None
    #     success, image1, image2 = cache['reader'].read2()
    #     if not success:
    #         return None
    #     image1 = cv2.resize(image1, (self._frame_width, self._frame_height), interpolation=cv2.INTER_AREA)
    #     image2 = cv2.resize(image2, (self._frame_width, self._frame_height), interpolation=cv2.INTER_AREA)
    #     frame1, frame2 = shared_frames[0], shared_frames[1]
    #     frame1[:] = image1
    #     frame2[:] = image2
    #     cache['current_count'] += 1
    #     self.logger.debug(f'{cache}')
    #     return msg.reset(cache['current_count'])

    def pre_loop(self, cache):# {{{
        if self._source_path is None or not os.path.isdir(self._source_path):
            self.send(None)
            self.logger.error(f'image dir path: {self._source_path} is not valid!!!')
            return

        class IMReader(object):
            def __init__(self, source, _from, _to):
                paths = []
                for entry in os.scandir(source):
                    if not entry.is_file():
                        continue
                    suffix = entry.name.split('.')[-1]
                    if suffix not in ('png', 'jpg', 'jpeg'):
                        continue
                    paths.append(entry.path)
                self._count = len(paths)
                if _from >= 1:
                    _from -= 1
                if _to <= 0:
                    self._image_list = iter(paths[_from:])
                else:
                    self._image_list = iter(paths[_from:_to])

            @property
            def count(self):
                return self._count

            def read(self):
                try:
                    img_path = next(self._image_list)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    return True, img
                except StopIteration:
                    return False, None

            # Only Debug
            def read2(self):
                try:
                    img = cv2.imread(next(self._image_list), cv2.IMREAD_COLOR)
                    return True, img[:,0:300], img[:, 300:]
                except StopIteration:
                    return False, None

            def release(self):
                pass

        imreader = IMReader(self._source_path, self._from_index, self._to_index)
        cache['reader'] = imreader
        cache['frame_count'] = imreader.count
        cache['frame_from'] = self._from_index
        cache['frame_stop'] = imreader.count if self._to_index < 0 else self._to_index
        cache['current_count'] = self._from_index
        super().pre_loop(cache)
        self.logger.info(f'pre cache: {cache}')# }}}

    def post_loop(self, cache):# {{{
        if 'reader' in cache:
            cache['reader'].release()
        super().post_loop(cache)# }}}
