#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file camera.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-11-08 17:24

import numpy as np
import cv2

from dataclasses import dataclass


@dataclass
class CameraProp(object):
    fl: float
    cx: float
    cy: float
    skew: float = 0.
    aspect: float = 1.


class CameraPose(object):
    '''
    This is the camera extrinsic pose. Here is not the linear transform matrix (Camera Pose Transform)
    '''

    @classmethod
    def from_axis_angle(cls, r, t):
        return CameraPose(cv2.Rodrigues(np.array(r))[0], np.array(t))

    @classmethod
    def from_axis_xyz(cls, degrees, order, offset):
        assert len(degrees) == len(order)
        rmat = np.identity(3)
        for degree, axis in zip(degrees, order):
            angle = np.radians(degree)
            s, c = np.sin(angle), np.cos(angle)
            if axis == 'x':
                rmat = rmat @ np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 'y':
                rmat = rmat @ np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            elif axis == 'z':
                rmat = rmat @ np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return CameraPose(rmat, offset).I

    def __init__(self, r, t):
        self.r = r
        self.t = np.asarray(t).reshape(3, 1)

    @property
    def I(self):  # noqa: E743
        '''
        http://assets.erlangai.cn/Misc/camera/Extrinsic_Matrix_from_Camera_Pose.png
        '''
        return CameraPose(self.r.T, - (self.r.T @ self.t))

    @property
    def E(self):
        '''
        Return: camera extrinsic matrix
        '''
        return np.hstack((self.r, self.t))

    def __matmul__(self, other):
        '''
        | Rl  Cl  |    | Rr  Cr  |   | Rl Rr  RlCr +_Cl |
        |         | @  |         | = |                  |
        | 0    1  |    | 0    1  |   |   0        1     |
        '''
        return CameraPose(self.r @ other.r, self.r @ other.t + self.t)

    def __repr__(self):
        return f'R:\n{self.r}\nT:\n{self.t}'


class PerspectiveCamera:

    def __init__(self, prop:CameraProp, pose:CameraPose):
        '''
        | fl  s   cx |
        | 0  a*fl cy |
        | 0   0    1 |
        '''
        self.K = np.asarray([[prop.fl, prop.skew, prop.cx], [0, prop.aspect * prop.fl, prop.cy], [0, 0, 1]])
        self.E = pose.E
        self.P = self.K @ self.E # 3 x 3 @ 3 x 4

    def project_3d(self, pts_3d: np.ndarray):
        '''
        pts_3d: N x 3
        pts_2d: N x 2 
        '''
        pts_3d_hg = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1)))) # N x 4
        pts_2d = pts_3d_hg @ self.P.T  # (P @ H^T)^T --> H @ P^T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def get_homography(self):
        """
        homography matrix from the projection matrix
        https://towardsdatascience.com/estimating-a-homography-matrix-522c70ec4b2c
        """
        return self.P[:, [0, 1, 3]]

    def __repr__(self):
        return f'K:\n{self.K}\nE:\n{self.E}\nP:\n{self.P}\n'
