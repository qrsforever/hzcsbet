#!/usr/bin/env python3


class SharedResult(object):
    def __init__(self, shmid: str, fwidth: int = 1920, fheight: int = 1080) -> None:
        self._shmid = shmid
        self._boxes_xyxy_list: list = []
        self._boxes_conf_list: list = []
        self._boxes_clas_list: list = []
        self._tracker_id_list: list = []
        self._team_color_list: list = []   # C.COLOR_CLUSTER_BOUNDARIES
        self._pitch_feat_list: list = []

        self._fwidth = fwidth
        self._fheigth = fheight
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
        outstr = '\n----------[ Cache ID: %s Token: %d Boxes Count: %d Shape: (%d %d) ]----------' % (
            self._shmid, self.token, boxcnt, self._fwidth, self._fheigth
        ) + outstr
        return outstr

    def reset(self, token):
        self._boxes_xyxy_list.clear()
        self._boxes_conf_list.clear()
        self._boxes_clas_list.clear()
        self._tracker_id_list.clear()
        self._team_color_list.clear()
        self._pitch_feat_list.clear()
        self._token = token

        return self

    @property
    def token_id(self):
        return self._token

    @property
    def shm_name(self):
        return self._shmid

    @property
    def frame_width(self):
        return self._fwidth

    @property
    def frame_height(self):
        return self._fheigth

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
    def boxes_conf(self):
        return self._boxes_conf_list

    @boxes_conf.setter
    def boxes_conf(self, confs):
        self._boxes_conf_list = confs

    @property
    def boxes_clas(self):
        return self._boxes_clas_list

    @boxes_clas.setter
    def boxes_clas(self, cls):
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
