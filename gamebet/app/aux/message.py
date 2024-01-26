#!/usr/bin/env python3

import numpy as np
from multiprocessing.shared_memory import SharedMemory


class SharedResult(object):
    def __init__(self, shmid: str, fwidth: int = 1920, fheight: int = 1080, frate: int = 30) -> None:
        self._shmid = shmid
        self._boxes_xyxy_list = None
        self._boxes_conf_list = None
        self._tracker_id_list = None
        self._team_color_list = None

        self._fwidth = fwidth
        self._fheigth = fheight
        self._frate = frate

    @property
    def frame_width(self):
        return self._fwidth

    @property
    def frame_height(self):
        return self._fheigth

    @property
    def frame_rate(self):
        return self._frate

    @property
    def boxes_xyxy(self):
        return self._boxes_xyxy_list

    @boxes_xyxy.setter
    def boxes_xyxy(self, xyxys):
        self._boxes_xyxy_list = xyxys

    @property
    def boxes_conf(self):
        return self._boxes_conf_list

    @boxes_conf.setter
    def boxes_conf(self, confs):
        self._boxes_conf_list = confs

    @property
    def tracker_id(self):
        return self._tracker_id_list

    @tracker_id.setter
    def tracker_id(self, ids):
        self._tracker_id_list = ids

    @property
    def team_color(self):
        return self._team_color_list

    @team_color.setter
    def team_color(self, colors):
        self._team_color_list = colors

    def get_frame(self):
        shm = SharedMemory(self._shmid, create=False)
        shape = (self._fheigth, self._fwidth, 3)
        return np.frombuffer(shm.buf[1:], dtype=np.uint8).reshape(shape)
