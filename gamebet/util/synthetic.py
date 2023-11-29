#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file synthetic.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-10 11:42

import cv2
import numpy as np
import scipy.io as sio
from collections import namedtuple
from dataclasses import dataclass

from .camera import CameraPose, CameraProp, PerspectiveCamera


Stats = namedtuple('Stats', ['mean', 'std', 'min', 'max'], defaults=[None] * 4)
Point = namedtuple('Point', ['x', 'y'])
MotionStd = namedtuple('MotionStd', ['fl', 'pan', 'tilt'], defaults=[0] * 3)
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

    def generate_camera_pair(motion_param:CameraMotionParameter, motion_std:MotionStd, image_size:tuple, camera_num:int):
        rnd_normal = lambda stat, size: np.random.normal(stat.mean, stat.std, (camera_num, size))
        rnd_uniform = lambda stat: np.random.uniform(stat.min, stat.max, camera_num)
        rnd_nearby = lambda d, std: d + np.random.uniform(-0.5, 0.5, 1) * std

        ccs = rnd_normal(motion_param.cc, 3)
        fls = rnd_normal(motion_param.fl, 1)
        rolls = rnd_uniform(motion_param.roll)
        pans = rnd_uniform(motion_param.pan)
        tilts = rnd_uniform(motion_param.tilt)

        fls_nb = rnd_nearby(fls, motion_std.fl)
        pans_nb = rnd_nearby(pans, motion_std.pan)
        tilts_nb = rnd_nearby(tilts, motion_std.tilt)

        cx, cy = image_size[0] / 2, image_size[1] / 2

        cams_pivot, cams_positive = np.zeros((camera_num, 9)), np.zeros((camera_num, 9))

        def _get_transform_matrix(axis_base_world, pan, tilt):
            axis_camera_base = CameraPose.from_axis_xyz([pan, tilt], order='yx', offset=[0, 0, 0])
            axis_camera_world = axis_camera_base @ axis_base_world
            return cv2.Rodrigues(axis_camera_world.r)[0], axis_camera_world.t.reshape(-1)

        for i in range(camera_num):
            axis_base_world = CameraPose.from_axis_xyz([-90, rolls[i], 0], order='xyz', offset=ccs[i])
            r, t = _get_transform_matrix(axis_base_world, pans[i], tilts[i])
            cams_pivot[i][0], cams_pivot[i][1], cams_pivot[i][2] = cx, cy, fls[i]
            cams_pivot[i][3], cams_pivot[i][4], cams_pivot[i][5] = r[0], r[1], r[2]
            cams_pivot[i][6], cams_pivot[i][7], cams_pivot[i][8] = t[0], t[1], t[2]

            r, t = _get_transform_matrix(axis_base_world, pans_nb[i], tilts_nb[i])
            cams_positive[i][0], cams_positive[i][1], cams_positive[i][2] = cx, cy, fls_nb[i]
            cams_positive[i][3], cams_positive[i][4], cams_positive[i][5] = r[0], r[1], r[2]
            cams_positive[i][6], cams_positive[i][7], cams_positive[i][8] = t[0], t[1], t[2]

        return cams_pivot, cams_positive

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


def generate_siamese_dataset(camera_parameter_file, soccer_field_template_file, output_file=None, image_size=(320, 180), image_num=10000):
    camera_param = sio.loadmat(camera_parameter_file)
    motion_param = CameraMotionParameter(
        cc = Stats(camera_param['cc_mean'][0], camera_param['cc_std'][0], camera_param['cc_min'][0], camera_param['cc_max'][0]),
        fl = Stats(camera_param['fl_mean'][0], camera_param['fl_std'][0], camera_param['fl_min'][0], camera_param['fl_max'][0]),
        pan = Stats(min=-35.0, max=35.0),
        roll = Stats(0, 0.2, -1.0, 1.0),
        tilt = Stats(min=-15.0, max=-5.0)
    )
    motion_std = MotionStd(30, 1.5, 0.75)

    ground_templ = sio.loadmat(soccer_field_template_file)
    soccer_field = SoccerField(ground_templ['points'], ground_templ['line_segment_index'])
    cams_pivot, cams_positive = SyntheticDataset.generate_camera_pair(motion_param, motion_std, image_size=(1280, 720), camera_num=image_num)

    image_piv_list, image_pos_list = [], []
    for i in range(image_num):
        image_piv = SyntheticDataset.generate_camera_image(cams_pivot[i], soccer_field, (1280, 720), thickness=4)
        image_pos = SyntheticDataset.generate_camera_image(cams_positive[i], soccer_field, (1280, 720), thickness=4)
        image_piv = cv2.cvtColor(cv2.resize(image_piv, image_size), cv2.COLOR_BGR2GRAY)
        image_pos = cv2.cvtColor(cv2.resize(image_pos, image_size), cv2.COLOR_BGR2GRAY)
        image_piv_list.append(image_piv)
        image_pos_list.append(image_pos)

    image_np = np.asarray(image_piv_list) / 255
    dataset = {
        'pivot_images': np.asarray(image_piv_list),
        'positive_images': np.asarray(image_pos_list),
        'pivot_camera': cams_pivot,
        'image_mean': np.mean(image_np),
        'image_std': np.std(image_np)
    }
    if output_file is not None:
        sio.savemat(output_file, dataset)
    return dataset

def generate_features_database(camera_parameter_file, soccer_field_template_file, image_size=(320, 180), image_num=10000):
    camera_param = sio.loadmat(camera_parameter_file)
    motion_param = CameraMotionParameter(
        cc = Stats(camera_param['cc_mean'][0], camera_param['cc_std'][0], camera_param['cc_min'][0], camera_param['cc_max'][0]),
        fl = Stats(camera_param['fl_mean'][0], camera_param['fl_std'][0], camera_param['fl_min'][0], camera_param['fl_max'][0]),
        pan = Stats(min=-35.0, max=35.0),
        roll = Stats(0, 0.2, -1.0, 1.0),
        tilt = Stats(min=-15.0, max=-5.0)
    )
    
    ground_templ = sio.loadmat(soccer_field_template_file)
    soccer_field = SoccerField(ground_templ['points'], ground_templ['line_segment_index'])
    cameras = SyntheticDataset.generate_ptz_cameras(motion_param, image_size=image_size, camera_num=image_num)
    
    image_list = []
    for i in range(image_num):
        image = SyntheticDataset.generate_camera_image(cameras[i], soccer_field, (1280, 720), thickness=4)
        image = cv2.cvtColor(cv2.resize(image, image_size), cv2.COLOR_BGR2GRAY)
        image_list.append(image)
        
    return {'images': np.asarray(image_list), 'cameras': cameras}
