#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file synthetic.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-10 11:42

import cv2
import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from .camera import CameraPose, CameraProp, PerspectiveCamera


Stats = namedtuple('Stats', ['mean', 'std', 'min', 'max'], defaults=[None] * 4)
Point = namedtuple('Point', ['x', 'y'])
SoccerField = namedtuple('SoccerField', ['points', 'line_segment_index'])


@dataclass
class CameraMotionParameter(object):
    cc: Stats
    fl: Stats
    pan: Stats
    roll: Stats
    tilt: Stats


class SyntheticDataset(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_ptz_cameras(motion_param:CameraMotionParameter, image_size:tuple, camera_num:int):
        rnd_normal = lambda stat, size: np.random.normal(stat.mean, stat.std, (camera_num, size))
        rnd_uniform = lambda stat: np.random.uniform(stat.min, stat.max, camera_num)
        ccs = rnd_normal(motion_param.cc, 3)
        fls = rnd_normal(motion_param.fl, 1)
        rolls = rnd_uniform(motion_param.roll)  # z-axis when in base axis, y-axis when in world axis
        pans = rnd_uniform(motion_param.pan)    # y-axis when in base axis
        tilts = rnd_uniform(motion_param.tilt)  # x-axis when in base axis
        cx, cy = image_size[0] / 2, image_size[1] / 2

        cameras = np.zeros((camera_num, 9))
        for i in range(camera_num):
            axis_base_world = CameraPose.from_axis_xyz([-90, rolls[i], 0], order='xyz', offset=ccs[i])
            axis_camera_base = CameraPose.from_axis_xyz([pans[i], tilts[i]], order='yx', offset=[0, 0, 0])
            axis_camera_world = axis_camera_base @ axis_base_world
            r = cv2.Rodrigues(axis_camera_world.r)[0]
            t = axis_camera_world.t
            cameras[i][0], cameras[i][1], cameras[i][2] = cx, cy, fls[i]
            cameras[i][3], cameras[i][4], cameras[i][5] = r[0], r[1], r[2]
            cameras[i][6], cameras[i][7], cameras[i][8] = t[0][0], t[1][0], t[2][0]
        return cameras

    @staticmethod
    def generate_camera_image(camera_data:np.ndarray, soccer_field:SoccerField, image_size:tuple, thickness=4):
        prop = CameraProp(cx=camera_data[0], cy=camera_data[1], fl=camera_data[2])
        pose = CameraPose.from_axis_angle(camera_data[3:6], camera_data[6:9])
        camera = PerspectiveCamera(prop, pose)

        image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        color = (255, 255, 255)
        points = soccer_field.points
        segmts = soccer_field.line_segment_index
        pts_3d = np.hstack((points, np.ones((points.shape[0], 1))))
        pts_2d = camera.project_3d(pts_3d)
        for idx1, idx2 in segmts:
            p1, p2 = np.rint(pts_2d[idx1]).astype(np.int32), np.rint(pts_2d[idx2]).astype(np.int32)
            cv2.line(image, tuple(p1), tuple(p2), color, thickness=thickness)
        return image
