#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np


class SinkingExecutor(ExecutorBase):
    _name = 'Sinking'

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        out_image = cache['draw_detections'](
            frame, msg.boxes_xyxy, msg.boxes_clas,
            msg.tracklet_ids, msg.team_color)
        cache['writer'].write(out_image)
        return msg

    def pre_loop(self, cache):
        import cv2
        import os
        video_path = os.environ.get('VIDEO_OUTPUT_PATH')
        cache['writer'] = cv2.VideoWriter(
            video_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=C.FRAME_RATE, frameSize=(C.FRAME_WIDTH, C.FRAME_HEIGHT), isColor=True)

        def draw_detections(frame, xyxy_list, clas_list, trid_list, color_list):
            image = frame.copy()
            if xyxy_list is None or len(xyxy_list) == 0:
                return image
            box_color = C.DETECTION_COLORS['player']
            draw_trid = False if trid_list is None else True
            draw_team = False if color_list is None else True
            for i in range(len(xyxy_list)):
                if clas_list[i] != 1: # players
                    continue
                x1, y1, x2, y2 = xyxy_list[i]
                if draw_team:
                    box_color = C.COLOR_CLUSTER_BOUNDARIES[color_list[i]][1]
                cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=box_color, thickness=3)
                if draw_trid:
                    cv2.putText(image, str(trid_list[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 3)
            return image

        cache['draw_detections'] = draw_detections

    def post_loop(self, cache):
        cache['writer'].release()
