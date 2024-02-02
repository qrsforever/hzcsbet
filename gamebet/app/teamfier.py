#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np


class TeamfierExecutor(ExecutorBase):
    _name = 'Teamfier'

    def run(self, shared_frames: tuple[np.ndarray], msg: SharedResult, cache: dict) -> SharedResult:
        frame = shared_frames[0]
        msg.team_colors = cache['detect_color'](frame, msg.boxes_xyxy)
        return msg

    def pre_loop(self, cache):
        import cv2

        def detect_color(frame, xyxy_list):
            color_list = []
            for i in range(len(xyxy_list)):
                x1, y1, x2, y2 = xyxy_list[i]
                image = frame[y1:y2, x1:x2]
                dst_color_index = 0
                max_nonzero_count = 0
                for i, (low_c, _, high_c) in enumerate(C.COLOR_CLUSTER_BOUNDARIES):
                    mask = cv2.inRange(image, np.array(low_c), np.array(high_c))
                    output = cv2.bitwise_and(image, image, mask=mask)
                    nonzero = np.count_nonzero(output)
                    if nonzero > max_nonzero_count:
                        max_nonzero_count = nonzero
                        dst_color_index = i
                color_list.append(dst_color_index)
            return color_list
        cache['detect_color'] = detect_color
