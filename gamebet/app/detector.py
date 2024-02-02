#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np


class DetectExecutor(ExecutorBase):
    _name = 'Detector'
    _weights_path = C.YOLO_DETECT_WEIGHTS_PATH

    def __init__(self, conf=0.15):
        super().__init__()
        self._conf = conf

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        frame = shared_frames[0]
        detections = cache['detect'](frame, conf=self._conf, verbose=False)[0]
        xyxy_list, conf_list, clas_list = [], [], []
        for pred in detections:
            xyxy_list.append(pred.boxes.xyxy.int().tolist()[0])
            conf_list.append(pred.boxes.conf.item())
            clas_list.append(pred.boxes.cls.item())
        msg.boxes_xyxy = xyxy_list
        msg.boxes_confs = conf_list
        msg.boxes_clses = clas_list
        return msg

    def pre_loop(self, cache):
        from ultralytics import YOLO
        cache['detect'] = YOLO(self._weights_path)
