#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import numpy as np


class DetectExecutor(ExecutorBase):
    _name = 'Detector'

    def __init__(self, weights_path, conf=0.15):
        super().__init__()
        self._weights_path = weights_path
        self._conf = conf

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        detections = cache['detect'](frame, conf=self._conf, verbose=False)[0]
        xyxy_list, conf_list, clas_list = [], [], []
        for pred in detections:
            xyxy_list.append(pred.boxes.xyxy.int().tolist()[0])
            conf_list.append(pred.boxes.conf.item())
            clas_list.append(pred.boxes.cls.item())
        msg.boxes_xyxy = xyxy_list
        msg.boxes_conf = conf_list
        msg.boxes_clas = clas_list
        return msg

    def pre_loop(self, cache):
        from ultralytics import YOLO
        cache['detect'] = YOLO(self._weights_path)
