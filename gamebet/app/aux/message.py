#!/usr/bin/env python3

import numpy as np


class SharedResult(object):
    def __init__(self, shmid: str, shmsz: int, fshape: tuple[int, int, int]) -> None:
        self._shmid = shmid
        self._shmsz = shmsz
        self._boxes_xyxy_list: list = []
        self._boxes_conf_list: list = []
        self._boxes_clas_list: list = []
        self._tracker_id_list: list = []
        self._team_color_list: list = []   # C.COLOR_CLUSTER_BOUNDARIES
        self._pitch_feat_list: list = []
        self._homography_matrix: list  = []

        self._fshape = fshape
        self._fsize = np.prod(fshape) * np.dtype('uint8').itemsize
        self._token = -1


    def __repr__(self) -> str:
        precnt, boxcnt, outstr = 20, 0, ''
        if len(self._boxes_xyxy_list) > 0:
            outstr += '\n  xyxy_list: %s' % self._boxes_xyxy_list[:precnt]
            boxcnt = len(self._boxes_xyxy_list)
        if len(self._boxes_conf_list) > 0:
            outstr += '\n  conf_list: %s' % self._boxes_conf_list[:precnt]
        if len(self._boxes_clas_list) > 0:
            outstr += '\n  clas_list: %s' % self._boxes_clas_list[:precnt]
        if len(self._tracker_id_list) > 0:
            outstr += '\n  trid_list: %s' % self._tracker_id_list[:precnt]
        if len(self._team_color_list) > 0:
            outstr += '\n  team_list: %s' % self._team_color_list[:precnt]
        if len(self._pitch_feat_list) > 0:
            outstr += '\n  feat_list: %s' % self._pitch_feat_list
        if len(self._homography_matrix) > 0:
            outstr += '\n  homo_list: %s' % self._homography_matrix
        outstr = '\n----------[ Cache ID: %s Token: %d Boxes Count: %d Shape: %s ]----------' % (
            self._shmid, self.token, boxcnt, self._fshape
        ) + outstr
        return outstr

    def reset(self, token):
        self._boxes_xyxy_list.clear()
        self._boxes_conf_list.clear()
        self._boxes_clas_list.clear()
        self._tracker_id_list.clear()
        self._team_color_list.clear()
        self._pitch_feat_list.clear()
        self._homography_matrix.clear()
        self._token = token

        return self

    @property
    def token_id(self):
        return self._token

    @property
    def shm_name(self):
        return self._shmid

    @property
    def shm_size(self):
        return self._shmsz

    @property
    def frame_shape(self):
        return self._fshape

    @property
    def frame_size(self):
        return self._fsize

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, token):
        self._token = token

    @property
    def boxes_xyxy(self):
        return self._boxes_xyxy_list

    @boxes_xyxy.setter
    def boxes_xyxy(self, xyxys):
        self._boxes_xyxy_list = xyxys

    @property
    def boxes_confs(self):
        return self._boxes_conf_list

    @boxes_confs.setter
    def boxes_confs(self, confs):
        self._boxes_conf_list = confs

    @property
    def boxes_clses(self):
        return self._boxes_clas_list

    @boxes_clses.setter
    def boxes_clses(self, cls):
        self._boxes_clas_list = cls

    @property
    def tracklet_ids(self):
        return self._tracker_id_list

    @tracklet_ids.setter
    def tracklet_ids(self, ids):
        self._tracker_id_list = ids

    @property
    def team_colors(self):
        return self._team_color_list

    @team_colors.setter
    def team_colors(self, colors):
        self._team_color_list = colors

    @property
    def pitch_feats(self):
        return self._pitch_feat_list

    @pitch_feats.setter
    def pitch_feats(self, feats):
        self._pitch_feat_list = feats

    @property
    def homography_matrix(self):
        return self._homography_matrix

    @homography_matrix.setter
    def homography_matrix(self, homo):
        self._homography_matrix = homo
