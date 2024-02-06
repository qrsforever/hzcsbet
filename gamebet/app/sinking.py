#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np
import cv2


class SinkingExecutor(ExecutorBase):
    _name = 'Sinking'

class FileSinkExecutor(ExecutorBase):

    def __init__(self, video_output_path, fps=25):
        super().__init__()
        self.video_output_path = video_output_path
        self.fps = fps

    def pre_loop(self, cache):
        cache['writer'] = cv2.VideoWriter(
            self.video_output_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps=self.fps, frameSize=(C.FRAME_WIDTH, C.FRAME_HEIGHT), isColor=True)

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

    def draw_birdeyes_perspective(self, frame, msg, bg_img=None):
        template_w, template_h = C.TEMPLATE_W, C.TEMPLATE_H
        render_w, render_h = C.RENDER_W, C.RENDER_H

        homo = np.array(msg.homography_matrix)
        if len(homo) == 0:
            return frame

        scaling_mat = np.eye(3)
        scaling_mat[0, 0] = render_w / template_w
        scaling_mat[1, 1] = render_h / template_h
        homo = scaling_mat @ homo

        def transform_matrix(matrix, p, vid_shape, gt_shape=()):
            p = (p[0]*C.FRAME_WIDTH/vid_shape[1], p[1]*C.FRAME_HEIGHT/vid_shape[0])
            px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
            py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
            p_after = (int(px*gt_shape[1]/C.RENDER_W) , int(py*gt_shape[0]/C.RENDER_H))
            return p_after

        xyxy_list, trid_list = msg.boxes_xyxy, msg.tracklet_ids,
        color_list, clas_list = msg.team_colors, msg.boxes_clses
        if bg_img is not None and len(xyxy_list) > 0 and len(trid_list) > 0 and len(color_list) > 0:
            for i in range(len(xyxy_list)):
                if clas_list[i] != 1: # players  # type: ignore
                    continue
                x1, y1, x2, y2 = xyxy_list[i]
                x_c, y_c = (x1 + x2) / 2, y2
                color = C.COLOR_CLUSTER_BOUNDARIES[color_list[i]][1]  # type: ignore
                coords = transform_matrix(homo, (x_c, y_c), frame.shape[:2], bg_img.shape[:2])
                cv2.circle(bg_img, coords, 12, color, -1)
                cv2.putText(bg_img, str(trid_list[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,), 3)

        image = cv2.warpPerspective(
            frame, homo, (render_w, render_h),
            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))  # pyright: ignore
        image = cv2.resize(image, frame.shape[:2][::-1])
        return image

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
        image1 = self.draw_detect_with_tracking(frame2, msg)
        bg_image = cache['bg_img'].copy()
        image2 = self.draw_birdeyes_perspective(frame1, msg, bg_img=bg_image)
        grid = np.vstack((np.hstack((frame1, image1)), np.hstack((image2, cache['bg_img']))))
        grid = cv2.resize(grid, frame1.shape[:2][::-1])
        cache['writer'].write(grid)
        return msg

    def pre_loop(self, cache):
        cache['writer'] = cv2.VideoWriter(
            self.video_output_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps=self.fps, frameSize=(C.FRAME_WIDTH, C.FRAME_HEIGHT), isColor=True)

        bg_img = cv2.imread(C.BG_BLACK_FIELD_PATH)
        bg_img = cv2.resize(bg_img, (C.FRAME_WIDTH, C.FRAME_HEIGHT))
        cache['bg_img'] = bg_img

