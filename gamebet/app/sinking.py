#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np
import cv2


class SinkingExecutor(ExecutorBase):
    _name = 'Sinking'

class FileSinkExecutor(ExecutorBase):

    def __init__(self, video_output_path):
        super().__init__()
        self.video_output_path = video_output_path

    def pre_loop(self, cache):
        cache['writer'] = cv2.VideoWriter(
            self.video_output_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps=C.FRAME_RATE, frameSize=(C.FRAME_WIDTH, C.FRAME_HEIGHT), isColor=True)

    def post_loop(self, cache):
        cache['writer'].release()

    def draw_detect_with_tracking(self, frame, msg):
        xyxy_list, clas_list = msg.boxes_xyxy, msg.boxes_clses
        trid_list, color_list = msg.tracklet_ids, msg.team_colors

        image = frame.copy()
        if len(xyxy_list) > 0:
            box_color = C.DETECTION_COLORS['player']
            draw_trid = True if len(trid_list) > 0 else False
            draw_team = True if len(color_list) > 0 else False
            for i in range(len(xyxy_list)):
                if clas_list[i] != 1: # players  # type: ignore
                    continue
                x1, y1, x2, y2 = xyxy_list[i]
                if draw_team:
                    box_color = C.COLOR_CLUSTER_BOUNDARIES[color_list[i]][1]  # type: ignore
                cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=box_color, thickness=3)
                if draw_trid:
                    cv2.putText(image, str(trid_list[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 3)
        return image

    def draw_birdeyes_perspective(self, frame, homo, scale=10, bg=0):
        yard2meter = 0.9144
        template_h, template_w = int(74 * yard2meter) + 2, int(115 * yard2meter) + 2

        # template view (0, 0) at top-left, so v-flip transform
        trans = np.array([
            [1, 0, 0],
            [0, -1, template_h],
            [0, 0, 1],
        ])
        if scale > 1:
            trans = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ]) @ trans
        h_cam2templ = np.linalg.inv(np.array(homo))
        image = cv2.warpPerspective(
            frame, trans @ h_cam2templ,
            (int(scale * template_w), int(scale * template_h)),
            borderMode=cv2.BORDER_CONSTANT, borderValue=bg)  # pyright: ignore
        return cv2.resize(image, frame.shape[:2][::-1])



class DirectSinkExecutor(FileSinkExecutor):
    _name = 'DirectSink'

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        frame = shared_frames[0]
        cache['writer'].write(frame)
        return msg


class BlendSinkExecutor(FileSinkExecutor):
    _name = 'BlendSink'

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        frame, view = shared_frames[0], shared_frames[1]
        xyxy_list, clas_list = msg.boxes_xyxy, msg.boxes_clses
        trid_list, color_list = msg.tracklet_ids, msg.team_colors
        feat_list = msg.pitch_feats

        image = frame.copy()
        if len(xyxy_list) > 0:
            box_color = C.DETECTION_COLORS['player']
            draw_trid = True if len(trid_list) > 0 else False
            draw_team = True if len(color_list) > 0 else False
            for i in range(len(xyxy_list)):
                if clas_list[i] != 1: # players  # type: ignore
                    continue
                x1, y1, x2, y2 = xyxy_list[i]
                if draw_team:
                    box_color = C.COLOR_CLUSTER_BOUNDARIES[color_list[i]][1]  # type: ignore
                cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=box_color, thickness=3)
                if draw_trid:
                    cv2.putText(image, str(trid_list[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 3)
        if len(feat_list) > 0:
            image[:] = cv2.addWeighted(image, 0.5, view, 0.5, 0)
        cache['writer'].write(image)
        return msg

class GridSinkExecutor(FileSinkExecutor):
    _name = 'GridSink'

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        frame1, frame2 = shared_frames[0], shared_frames[1]
        image1 = self.draw_detect_with_tracking(frame1, msg)
        image2 = self.draw_birdeyes_perspective(frame2, msg.homography_matrix)
        grid = np.vstack((np.hstack((frame1, image1)), np.hstack((frame2, image2))))
        grid = cv2.resize(grid, frame1.shape[:2][::-1])
        cache['writer'].write(grid)
        return msg

    def pre_loop(self, cache):
        cache['writer'] = cv2.VideoWriter(
            self.video_output_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps=5, frameSize=(C.FRAME_WIDTH, C.FRAME_HEIGHT), isColor=True)
