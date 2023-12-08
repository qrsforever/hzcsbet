#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


get_ipython().system('$PIP_INSTALL pyflann-py3 torchviz torchsummary')


# In[2]:


get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('watermark', '-p pytransform3d,matplotlib,scipy,numpy,pyflann')
# %watermark -p numpy,sklearn,pandas
# %watermark -p ipywidgets,cv2,PIL,matplotlib,plotly,netron
# %watermark -p torch,torchvision,torchaudio
# %watermark -p tensorflow,tensorboard,tflite
# %watermark -p onnx,tf2onnx,onnxruntime,tensorrt,tvm
# %matplotlib inline
# %config InlineBackend.figure_format='retina'
# %config IPCompleter.use_jedi = False

# %matplotlib inline
# %matplotlib widget
# from IPython.display import display, Markdown, HTML, IFrame, Image, Javascript
# from IPython.core.magic import register_line_cell_magic, register_line_magic, register_cell_magic
# display(HTML('<style>.container { width:%d%% !important; }</style>' % 90))

import sys, os, io, logging, time, random, math
import json, base64, requests, shutil
import argparse, shlex, signal
import numpy as np
import scipy.io as sio
import torch

np.set_printoptions(
    edgeitems=3, infstr='inf',
    linewidth=75, nanstr='nan', precision=6,
    suppress=True, threshold=100, formatter=None)

argparse.ArgumentParser.exit = lambda *arg, **kwargs: _IGNORE_

def _IMPORT(x, tag='main', debug=False):
    def __request_text(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise RuntimeError(url)
    try:
        x = x.strip()
        if x[0] == '/' or x[1] == '/':
            with open(x) as fr:
                x = fr.read()
        elif 'github' in x or 'gitee' in x:
            if x.startswith('import '):
                x = x[7:]
            if x.startswith('https://'):
                x = x[8:]
            if not x.endswith('.py'):
                x = x + '.py'
            x = x.replace('blob/main/', '').replace('blob/master/', '')
            if x.startswith('raw.githubusercontent.com'):
                x = 'https://' + x
                x = __request_text(x)
            elif x.startswith('github.com'):
                x = x.replace('github.com', 'raw.githubusercontent.com')
                mod = x.split('/')
                x = 'https://' + '/'.join(mod[:3]) + f'/{tag}/' + '/'.join(mod[-3:])
                x = __request_text(x)
            elif x.startswith('gitee.com'):
                mod = x.split('/')
                x = 'https://' + '/'.join(mod[:3]) + f'/raw/{tag}/' + '/'.join(mod[3:])
                x = __request_text(x)
        if debug:
            return x
        else:
            exec(x, globals())
    except Exception as err:
        # sys.stderr.write(f'request {x} : {err}')
       pass

def _DIR(x, dumps=True, ret=True):
    attrs = sorted([y for y in dir(x) if not y.startswith('_')])
    result = '%s: %s' % (str(type(x))[8:-2], json.dumps(attrs) if dumps else attrs)
    if ret:
        return result
    print(result)


###
### Display ###
###

_IMPORT('import pandas as pd')
_IMPORT('import cv2')
_IMPORT('from PIL import Image')
_IMPORT('import matplotlib.pyplot as plt')
# _IMPORT('import plotly')
# _IMPORT('import plotly.graph_objects as go')
# _IMPORT('import ipywidgets as widgets')
# _IMPORT('from ipywidgets import interact, interactive, fixed, interact_manual')
_IMPORT('import pytransform3d.rotations')
_IMPORT('import pytransform3d.camera')
_IMPORT('import pytransform3d.transformations')
_IMPORT('import pyflann')
# plotly.offline.init_notebook_mode(connected=False)

plt.rcParams['figure.figsize'] = (12.0, 8.0)
# from matplotlib.font_manager import FontProperties
# simsun = FontProperties(fname='/sysfonts/simsun.ttc', size=12)

# _IMPORT('gitee.com/qrsforever/nb_easy/easy_widget')

def nbeasy_make_grid(images, nrow=None, padding=4, pad_value=127, labels=None,
                     font_scale=1.0, font_thickness=1, text_color=(255,), text_color_bg=None):
    count = len(images)
    if isinstance(images, dict):
        labels = [lab for lab in images.keys()]
        images = [img for img in images.values()]

    if not isinstance(images, (list, tuple, np.ndarray)) or count == 0 or not isinstance(images[0], np.ndarray):
        return
    if nrow is None or nrow > count:
        nrow = count

    max_h, max_w = np.asarray([img.shape[:2] for img in images]).max(axis=0)
    if labels is not None:
        text_org = int(0.1 * max_w), int(0.9 * max_h)
        shape_length = 3
    else:
        shape_length = np.asarray([len(img.shape) for img in images]).max()
    lack = count % nrow
    rows = np.int0(np.ceil(count / nrow))
    hpad_size = [max_h, padding] 
    if rows > 1:
        vpad_size = [padding, nrow * max_w + (nrow - 1) * padding]
        if lack > 0:
            lack_size = [max_h, max_w]
    if shape_length == 3:
        hpad_size.append(3)
        if rows > 1:
            vpad_size.append(3)
            if lack > 0:
                lack_size.append(3)
    hpadding = pad_value * np.ones(hpad_size, dtype=np.uint8)
    if rows > 1:
        vpadding = pad_value * np.ones(vpad_size, dtype=np.uint8)
        if lack > 0:
            lack_image = pad_value * np.ones(lack_size, dtype=np.uint8)
            images.extend([lack_image] * lack)
            if labels is not None:
                labels.extend([''] * lack)
    vlist = []
    for i in range(rows):
        hlist = []
        for j in range(nrow):
            if j != 0:
                hlist.append(hpadding)
            timg = images[i * nrow + j].copy()
            th, tw = timg.shape[:2]
            if th != max_h or tw != max_w:
                timg = cv2.resize(timg, (max_w, max_h))
            if len(timg.shape) != shape_length:
                timg = cv2.cvtColor(timg, cv2.COLOR_GRAY2BGR)
            if labels is not None:
                text = str(labels[i * nrow + j])
                if len(text) > 0:
                    if text_color_bg is not None:
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        pos1 = text_org[0] - int(font_scale * 5), text_org[1] - th - int(font_scale * 5)
                        pos2 = text_org[0] + int(font_scale * 5) + tw, text_org[1] + int(font_scale * 8)
                        cv2.rectangle(timg, pos1, pos2, text_color_bg, -1)
                    cv2.putText(timg, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            hlist.append(timg)
        if i != 0:
            vlist.append(vpadding)
        vlist.append(np.hstack(hlist))
    if rows > 1:
        return np.vstack(vlist)
    return vlist[0]

def nbeasy_imshow(image, title=None, color='bgr', figsize=(6, 3), canvas=False):
    import IPython
    plt.close('all')
    if figsize == 'auto':
        ih, iw = image.shape[:2]
        fw, fh = int(1.5 * iw / 80) + 1, int(1.5 * ih / 80) + 1
        if fw > 32:
            fh = int(32 * (fh / fw))
            fw = 32
        figsize = (fw, fh)
    if canvas:
        IPython.get_ipython().enable_matplotlib(gui='widget');
        fig = plt.figure(figsize=figsize, dpi=80)
        fig.canvas.toolbar_position = 'left'
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = True
    else:
        IPython.get_ipython().enable_matplotlib(gui='inline')
        fig = plt.figure(figsize=figsize, dpi=80)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if color == 'gray' or len(image.shape) == 2:
        plt.imshow(image, cmap='gray');
    else:
        if color == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image);
        
_imgrid = nbeasy_make_grid
_imshow = nbeasy_imshow


# In[ ]:





# In[3]:


from collections import namedtuple
from dataclasses import dataclass


# ## Global

# In[4]:


TOP_DIR='/jupyter/hzcsbet/gamebet'
CKPTS_DIR = f'{TOP_DIR}/checkpoints'
HEIGHT, WIDTH = 720, 1280
yard2meter = 0.9144
template_h, template_w = int(74 * yard2meter) + 2, int(115 * yard2meter) + 2
interpolation = cv2.INTER_AREA # 缩小适用

class COLORS(object):
    # BGR
    GREEN      = (0   , 255 , 0)
    RED        = (0   , 0   , 255)
    BLACK      = (0   , 0   , 0)
    YELLOW     = (0   , 255 , 255)
    WHITE      = (255 , 255 , 255)
    CYAN       = (255 , 255 , 0)
    MAGENTA    = (255 , 0   , 242)
    GOLDEN     = (32  , 218 , 165)
    LIGHT_BLUE = (255 , 9   , 2)
    PURPLE     = (128 , 0   , 128)
    CHOCOLATE  = (30  , 105 , 210)
    PINK       = (147 , 20  , 255)
    ORANGE     = (0   , 69  , 255)

sys.path.append(f'{TOP_DIR}/python')
    
if not os.path.isdir(CKPTS_DIR):
    os.mkdir(CKPTS_DIR)


# In[5]:


def _print_dictkeys_shape(d):
    for k, v in d.items():
        if k.startswith('__'):
            continue
        print(f'{k}: {v.shape}')
        
def _print_statistics(o, prefix=''):
    if isinstance(o, np.ndarray):
        print(f'{prefix} mean: {o.mean()}, std: {o.std()}, max: {o.max()}, min:{o.min()}')
    
def _print_matrix(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
_print_star = lambda n = 50: print('*' * n)


# In[6]:


def _warp_perspective(h_templ2cam, img_cam, img_size, scale=1, bg=0):
    # template view (0, 0) at top-left, so v-flip transform 
    trans = np.array([
        [1, 0, 0],
        [0, -1, img_size[1]],
        [0, 0, 1],
    ])
    if scale > 1:
        trans = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ]) @ trans
    h_cam2templ = np.linalg.inv(h_templ2cam)
    return cv2.warpPerspective(img_cam, trans @ h_cam2templ, (scale * img_size[0], scale * img_size[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


# ## Dataset

# ### worldcup2014.mat

# In[7]:


worldcup_2014_mat = sio.loadmat(f'{TOP_DIR}/data/worldcup2014.mat')
_print_dictkeys_shape(worldcup_2014_mat)


# In[8]:


points = worldcup_2014_mat['points']
_print_statistics(points[:, 0], 'points'), _print_statistics(points[:, 1], 'points');


#     UofT:
#         template_h: 74 * 0.9144 = 67.6656
#         template_w: 115 * 0.9144 = 105.156

# In[9]:


points[100:110]


# In[10]:


worldcup_2014_mat['line_segment_index'][:10]


# In[11]:


w, h = int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max()))
img_soccer_field = np.zeros((h, w, 3), dtype=np.uint8)
for idx1, idx2 in worldcup_2014_mat['line_segment_index']:
    p1, p2 = points[idx1], points[idx2]
    q1 = np.rint(p1).astype(np.int32)
    q2 = np.rint(p2).astype(np.int32)
    cv2.line(img_soccer_field, tuple(q1), tuple(q2), COLORS.WHITE)
_imshow(img_soccer_field, title='Soccer Field', canvas=False)
img_soccer_field.shape


# ### worldcup_dataset_camera_parameter.mat

# In[12]:


worldcup_camera_parameter = sio.loadmat(f'{TOP_DIR}/data/worldcup_dataset_camera_parameter.mat')
_print_dictkeys_shape(worldcup_camera_parameter)


# In[13]:


worldcup_camera_parameter


# ### worldcup_sampled_cameras.mat

# In[14]:


worldcup_sampled_cameras = sio.loadmat(f'{TOP_DIR}/data/worldcup_sampled_cameras.mat')
_print_dictkeys_shape(worldcup_sampled_cameras)


# In[15]:


worldcup_sampled_cameras['pivot_cameras'][12:18]


# In[16]:


worldcup_sampled_cameras['positive_ious'][12:18]


# ### features/database_camera_feature.mat

# In[17]:


database_camera_features = sio.loadmat(f'{TOP_DIR}/data/features/database_camera_feature.mat')
_print_dictkeys_shape(database_camera_features)


# In[18]:


database_camera_features['cameras'][:5]


# In[19]:


# calculate from siamese neural network
database_camera_features['features'][:5]


# ### features/testset_feature.mat

# In[20]:


testset_features = sio.loadmat(f'{TOP_DIR}/data/features/testset_feature.mat')
_print_dictkeys_shape(testset_features)


# In[21]:


# calculate by cv.distanceTransform
show_num = 5
edge_distances_x = testset_features["edge_distances"][..., 12:12 + show_num]
edge_distances_x.shape


# In[22]:


_imshow(np.hstack(np.split(edge_distances_x, indices_or_sections=show_num, axis=-1)).squeeze(), color='gray', figsize=(3 * show_num, 2))


# In[23]:


edge_map_x = testset_features["edge_map"][..., 12:12 + show_num]
edge_map_x.shape


# In[24]:


_imshow(np.hstack(np.split(edge_map_x, indices_or_sections=show_num, axis=-1)).squeeze(), color='gray', figsize=(3 * show_num, 2))


# ### UoT_soccer/train_val.mat

# In[25]:


train_val_UoT = sio.loadmat(f'{TOP_DIR}/data/UoT_soccer/train_val.mat')
_print_dictkeys_shape(train_val_UoT)


# In[26]:


train_val_UoT['meta']


# In[27]:


type(train_val_UoT['annotation'][0][0]), train_val_UoT['annotation'][0][0]


# In[28]:


train_val_UoT['annotation'][0][15:17]


# In[29]:


h16_gt_mat = sio.loadmat(f'{TOP_DIR}/data/UoT_soccer/16_grass_gt.mat')
_print_dictkeys_shape(h16_gt_mat)


# In[30]:


nbeasy_imshow(h16_gt_mat['grass'], 'gray', figsize=(6, 3))


# ### UoT_soccer/test.mat (same size with testset_feature.mat)

# In[31]:


test_UoT = sio.loadmat(f'{TOP_DIR}/data/UoT_soccer/test.mat')
_print_dictkeys_shape(test_UoT)


# In[32]:


test_UoT['annotation'][0][14:17]


# In[33]:


get_ipython().system('cat $TOP_DIR/data/UoT_soccer/16.homographyMatrix')


# ## Utils (SCCvSD)
# 
# 
# $$
# (A @ B)^T = B^T @ A^T
# $$

# ### RotationUtil

# In[34]:


class RotationUtil:
    @staticmethod
    def rotate_x_axis(angle, T=True):
        """
        rotate coordinate with X axis
        https://en.wikipedia.org/wiki/Rotation_matrix + transpose
        http://mathworld.wolfram.com/RotationMatrix.html
        :param angle: in degree
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        if T:
            r = np.transpose(r)
        return r

    @staticmethod
    def rotate_y_axis(angle, T=True):
        """
        rotate coordinate with X axis
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        if T:
            r = np.transpose(r)
        return r

    @staticmethod
    def rotate_z_axis(angle, T=True):
        """

        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        if T:
            r = np.transpose(r)
        return r

    @staticmethod
    def pan_y_tilt_x(pan, tilt):
        """
        Rotation matrix of first pan, then tilt
        :param pan:
        :param tilt:
        :return:
        """
        r_tilt = RotationUtil.rotate_x_axis(tilt)
        r_pan = RotationUtil.rotate_y_axis(pan)
        m = r_tilt @ r_pan
        return m

    #  rot_vec, _ = cv.Rodrigues(rotation)
    @staticmethod
    def rotation_matrix_to_Rodrigues(m):
        assert m.shape[0] == 3 and m.shape[1] == 3
        rot_vec, _ = cv2.Rodrigues(m)
        return rot_vec


# In[35]:


degreex, degreey, degreez = 40, 70, -35
anglex, angley, anglez = math.radians(degreex), math.radians(degreey), math.radians(degreez)
anglex, angley, anglez


# In[36]:


rx = RotationUtil.rotate_x_axis(degreex, T=False)
ry = RotationUtil.rotate_y_axis(degreey, T=False)
rz = RotationUtil.rotate_z_axis(degreez, T=False)
rx, ry, rz


# In[37]:


rx_ = cv2.Rodrigues(np.array([anglex, 0, 0]))[0]
ry_ = cv2.Rodrigues(np.array([0, angley, 0]))[0]
rz_ = cv2.Rodrigues(np.array([0, 0, anglez]))[0]
rx_, ry_, rz_


# #### Be Carefull
# 
# 旋转向量与XYZ角度不是同一概念
# 
# [绕着旋转向量旋转一定的角度, 可以从一个位置变换到另一个位置](https://euclideanspace.com/maths/geometry/rotations/axisAngle/index.htm)

# In[38]:


rxyz = cv2.Rodrigues(np.array([anglex, angley, anglez]))[0]
rxyz


# In[39]:


rx @ ry @ rz, rz @ ry @ rx


# ### ProjectiveCamera

# In[40]:


class ProjectiveCamera:
    def __init__(self, fl, u, v, cc, rod_rot):
        """
        :param fl:
        :param u:
        :param v:
        :param cc:
        :param rod_rot:
        """
        self.K = np.zeros((3, 3)) # calibration matrix
        self.camera_center = np.zeros(3)
        self.rotation = np.zeros(3)

        self.P = np.zeros((3, 4)) # projection matrix

        self.set_calibration(fl, u, v)
        self.set_camera_center(cc)
        self.set_rotation(rod_rot)

    def set_calibration(self, fl, u, v):
        """
        :param fl:
        :param u:
        :param v:
        :return:
        """
        self.K = np.asarray([[fl, 0, u],
                             [0, fl, v],
                             [0, 0, 1]])
        self._recompute_matrix()

    def set_camera_center(self, cc):
        assert cc.shape[0] == 3

        self.camera_center[0] = cc[0]
        self.camera_center[1] = cc[1]
        self.camera_center[2] = cc[2]
        self._recompute_matrix()

    def set_rotation(self, rod_rot):
        """
        :param rod_rot: Rodrigues vector
        :return:
        """
        assert rod_rot.shape[0] == 3

        self.rotation[0] = rod_rot[0]
        self.rotation[1] = rod_rot[1]
        self.rotation[2] = rod_rot[2]
        self._recompute_matrix()

    def project_3d(self, x, y, z, w=1.0):
        """
        :param x:
        :param y:
        :param z:
        :return:
        """
        p = np.zeros(4)
        p[0],p[1],p[2], p[3] = x, y, z, w
        q = self.P @ p
        assert q[2] != 0.0
        return (q[0]/q[2], q[1]/q[2])

    def get_homography(self):
        """
        homography matrix from the projection matrix
        :return:
        """
        h = self.P[:, [0, 1,3]]
        return h


    def _recompute_matrix(self):
        """
        :return:
        """
        P = np.zeros((3, 4))
        for i in range(3):
            P[i][i] = 1.0

        for i in range(3):
            P[i][3] = -self.camera_center[i]

        r, _ = cv2.Rodrigues(self.rotation)
        #print(r)
        #print('{} {} {}'.format(self.K.shape, r.shape, P.shape))
        self.P = self.K @ r @ P


# ### SyntheticUtil

# In[41]:


class SyntheticUtil:
    @staticmethod
    def camera_to_edge_image(camera_data,
                             model_points, model_line_segment,
                             im_h=720, im_w=1280, line_width=4):
        """
         Project (line) model images using the camera
        :param camera_data: 9 numbers
        :param model_points:
        :param model_line_segment:
        :param im_h: 720
        :param im_w: 1280
        :return: H * W * 3 Opencv image
        """
        assert camera_data.shape[0] == 9

        u, v, fl = camera_data[0:3]
        rod_rot = camera_data[3:6]
        cc = camera_data[6:9]

        camera = ProjectiveCamera(fl, u, v, cc, rod_rot)
        im = np.zeros((im_h, im_w, 3), dtype=np.uint8)
        n = model_line_segment.shape[0]
        color = (255, 255, 255)
        for i in range(n):
            idx1, idx2 = model_line_segment[i][0], model_line_segment[i][1]
            p1, p2 = model_points[idx1], model_points[idx2]
            q1 = camera.project_3d(p1[0], p1[1], 0.0, 1.0)
            q2 = camera.project_3d(p2[0], p2[1], 0.0, 1.0)
            q1 = np.rint(q1).astype(np.int32)
            q2 = np.rint(q2).astype(np.int32)
            cv2.line(im, tuple(q1), tuple(q2), color, thickness=line_width)
        return im

    @staticmethod
    def distance_transform(img):
        """
        :param img: Opencv Image
        :return:
        """
        h, w, c = img.shape
        if c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            assert c == 1

        _, binary_im = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

        dist_im = cv2.distanceTransform(
            binary_im, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return dist_im

    @staticmethod
    def find_transform(im_src, im_dst):
        warp = np.eye(3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        # suggested by felibol
        # https://github.com/lood339/SCcv2SD/issues/8
        try:
            _, warp = cv2.findTransformECC(
                im_src, im_dst, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
        except:
            print('Warning: find transform failed. Set warp as identity')
        return warp

    @staticmethod
    def generate_ptz_cameras(cc_statistics,
                             fl_statistics,
                             roll_statistics,
                             pan_range, tilt_range,
                             u, v,
                             camera_num):
        """
        Input: PTZ camera base information
        Output: randomly sampled camera parameters
        :param cc_statistics:
        :param fl_statistics:
        :param roll_statistics:
        :param pan_range:
        :param tilt_range:
        :param u:
        :param v:
        :param camera_num:
        :return: N * 9 cameras
        """
        cc_mean, cc_std, cc_min, cc_max = cc_statistics
        fl_mean, fl_std, fl_min, fl_max = fl_statistics
        roll_mean, roll_std, roll_min, roll_max = roll_statistics
        pan_min, pan_max = pan_range
        tilt_min, tilt_max = tilt_range

        # generate random values from the distribution
        camera_centers = np.random.normal(cc_mean, cc_std, (camera_num, 3))
        focal_lengths = np.random.normal(fl_mean, fl_std, (camera_num, 1))
        rolls = np.random.normal(roll_mean, roll_std, (camera_num, 1))
        pans = np.random.uniform(pan_min, pan_max, camera_num)
        tilts = np.random.uniform(tilt_min, tilt_max, camera_num)

        cameras = np.zeros((camera_num, 9))
        for i in range(camera_num):
            base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_z_axis(rolls[i]) @                RotationUtil.rotate_x_axis(-90)
            pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pans[i], tilts[i])
            rotation = pan_tilt_rotation @ base_rotation
            rot_vec, _ = cv2.Rodrigues(rotation)

            cameras[i][0], cameras[i][1] = u, v
            cameras[i][2] = focal_lengths[i]
            cameras[i][3], cameras[i][4], cameras[i][5] = rot_vec[0], rot_vec[1], rot_vec[2]
            cameras[i][6], cameras[i][7], cameras[i][8] = camera_centers[i][0], camera_centers[i][1], camera_centers[i][2]
        return cameras

    @staticmethod
    def sample_positive_pair(pp, cc, base_roll, pan, tilt, fl,
                             pan_std, tilt_std, fl_std):
        """
        Sample a camera that has similar pan-tilt-zoom with (pan, tilt, fl).
        The pair of the camera will be used a positive-pair in the training
        :param pp: [u, v]
        :param cc: camera center
        :param base_roll: camera base, roll angle
        :param pan:
        :param tilt:
        :param fl:
        :param pan_std:
        :param tilt_std:
        :param fl_std:
        :return:
        """

        def get_nearby_data(d, std):
            assert std > 0
            delta = np.random.uniform(-0.5, 0.5, 1) * std
            return d + delta

        pan = get_nearby_data(pan, pan_std)
        tilt = get_nearby_data(tilt, tilt_std)
        fl = get_nearby_data(fl, fl_std)

        camera = np.zeros(9)
        camera[0] = pp[0]
        camera[1] = pp[1]
        camera[2] = fl

        base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_y_axis(base_roll) @            RotationUtil.rotate_x_axis(-90)
        pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pan, tilt)
        rotation = pan_tilt_rotation @ base_rotation
        rot_vec = RotationUtil.rotation_matrix_to_Rodrigues(rotation)
        camera[3: 6] = rot_vec.squeeze()
        camera[6: 9] = cc
        return camera

    @staticmethod
    def generate_database_images(pivot_cameras, positive_cameras,
                                 model_points, model_line_segment):
        """
        Default size 180 x 320 (h x w)
        Generate database image for siamese network training
        :param pivot_cameras:
        :param positive_cameras:
        :return:
        """
        n = pivot_cameras.shape[0]
        assert n == positive_cameras.shape[0]

        # N x 1 x H x W pivot images
        # N x 1 x H x w positive image
        # negative pairs are randomly selected
        im_h, im_w = 180, 320
        pivot_images = np.zeros((n, 1, im_h, im_w), dtype=np.uint8)
        positive_images = np.zeros((n, 1, im_h, im_w), dtype=np.uint8)

        for i in range(n):
            piv_cam = pivot_cameras[i, :]
            pos_cam = positive_cameras[i, :]

            piv_im = SyntheticUtil.camera_to_edge_image(
                piv_cam, model_points, model_line_segment, 720, 1280, 4)
            pos_im = SyntheticUtil.camera_to_edge_image(
                pos_cam, model_points, model_line_segment, 720, 1280, 4)

            # to a smaller image
            piv_im = cv2.resize(piv_im, (im_w, im_h))
            pos_im = cv2.resize(pos_im, (im_w, im_h))

            # to a gray image
            piv_im = cv2.cvtColor(piv_im, cv2.COLOR_BGR2GRAY)
            pos_im = cv2.cvtColor(pos_im, cv2.COLOR_RGB2GRAY)

            pivot_images[i, 0, :, :] = piv_im
            positive_images[i, 0, :, :] = pos_im

        return (pivot_images, positive_images)


# #### generate ptz camera

# In[42]:


data = worldcup_camera_parameter

cc_mean = data['cc_mean']
cc_std = data['cc_std']
cc_min = data['cc_min']
cc_max = data['cc_max']
cc_statistics = [cc_mean, cc_std, cc_min, cc_max]

fl_mean = data['fl_mean']
fl_std = data['fl_std']
fl_min = data['fl_min']
fl_max = data['fl_max']
fl_statistics = [fl_mean, fl_std, fl_min, fl_max]
roll_statistics = [0, 0.2, -1.0, 1.0]

pan_range = [-35.0, 35.0]
tilt_range = [-15.0, -5.0]
num_camera = 1

cameras = SyntheticUtil.generate_ptz_cameras(cc_statistics,
                                             fl_statistics,
                                             roll_statistics,
                                             pan_range, tilt_range,
                                             1280/2.0, 720/2.0,
                                             num_camera)


# In[43]:


cameras[0]


# #### generate edge camera

# In[44]:


data = worldcup_2014_mat
model_points = data['points']
model_line_index = data['line_segment_index']
im = SyntheticUtil.camera_to_edge_image(cameras[0], model_points, model_line_index, 720, 1280, line_width=4)
_imshow(im, figsize=(8, 4)), im.shape


# #### projective camera and homography 

# In[45]:


data = cameras[0]
camera = ProjectiveCamera(data[2], data[0], data[1], data[6:9], data[3:6])
h = camera.get_homography()
h


# In[46]:


warped_im = _warp_perspective(h, im, (template_w, template_h), scale=10)
_imshow(warped_im, figsize=(8, 4)), warped_im.shape


# ## Plot3D

# In[47]:


worldcup_camera_parameter


# In[48]:


img_soccer = cv2.cvtColor(img_soccer_field, cv2.COLOR_BGR2RGB)
img_soccer = np.transpose(img_soccer, (1, 0, 2))

plt.close('all')
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set(xlim=(-5, img_soccer.shape[0] + 5), ylim=(-45, img_soccer.shape[1] + 5), zlim=(0, 20))
ax = pytransform3d.rotations.plot_basis(
    ax,
    s=10, # 基坐标大小
    lw=3,
)

Rx_neg_90 = pytransform3d.rotations.active_matrix_from_angle(0, np.deg2rad(-90))
Ry_pos_00 = pytransform3d.rotations.active_matrix_from_angle(1, np.deg2rad(0))
Rz_pos_05 = pytransform3d.rotations.active_matrix_from_angle(2, np.deg2rad(5))
# R = pytransform3d.rotations.active_matrix_from_extrinsic_euler_xyz((np.deg2rad(-90), 0, np.deg2rad(5)))
R_c = Rz_pos_05 @ Ry_pos_00 @ Rx_neg_90 
C = worldcup_camera_parameter['cc_mean'][0]
ax = pytransform3d.rotations.plot_basis(
    ax,
    s=10, lw=3,
    R=R_c,
    p=C,
    # label='Camera PTZ'
)
xx, yy = np.ogrid[0:img_soccer.shape[0], 0:img_soccer.shape[1]]
# 设置x,y,z坐标轴刻度等大小
ax.set_box_aspect((np.ptp(xx), np.ptp(yy), 20))
ax.plot_surface(xx, yy, np.atleast_2d(0), alpha=0.75, rstride=1, cstride=1, facecolors=img_soccer.astype(np.float32) / 255.0)

# Camera
f = worldcup_camera_parameter['fl_mean'][0][0]
sensor_size = (1280, 720)
intrinsic_matrix = np.array([
    [f, 0, sensor_size[0] / 2],
    [0, f, sensor_size[1] / 2],
    [0, 0, 1.0]
], dtype=np.float32)

# H(p) @ H(R)
cam2world = pytransform3d.transformations.transform_from(R=R_c, p=C)
pytransform3d.camera.plot_camera(
    ax=ax,
    cam2world=cam2world,
    M=intrinsic_matrix, 
    sensor_size=sensor_size, # TODO
    virtual_image_distance=30,
    alpha=0.45, color='#181818'
)
ax.view_init(elev=30, azim=40, vertical_axis='z')
ax.set_title("camera transformation")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
R_c, C


# ### Camera Pose
# 
# Let $C$ be a column vector describing the location of the camera-center in world coordinates, and let $R_c$
# be the rotation matrix describing the camera's orientation with respect to the world coordinate axes.

# In[49]:


R_h = np.eye(4)
R_h[:3, :3] = R_c
R_h


# In[50]:


C_h = np.eye(4)
C_h[:3, -1] = C
C_h


# In[51]:


Pose = C_h @ R_h
Pose


# In[52]:


Pose.T, np.linalg.inv(Pose)


# ### Camera Extrinsic Pose
# 
# Describes how to transform points in world coordinates to camera coordinates. The vector $t$ can be interpreted as the position of the world origin in camera coordinates, and the columns of $R$ represent represent the directions of the world-axes in camera coordinates.
# 
# The extrinsic matrix is obtained by inverting the camera's pose matrix.

# In[53]:


R = R_c.T
t = -R_c.T @ C
R, t


# ## Search Feature

# In[54]:


query_index = 15
struct_void = test_UoT['annotation'][0][query_index]
type(struct_void), struct_void['image_name'], struct_void['homography']


# In[55]:


# ground truth homography
test_image_name, test_gt_h = struct_void
test_image_name, test_gt_h


# In[56]:


testset_features['features'].shape, database_camera_features['features'].shape


# In[57]:


database_features, testset_feature_x = database_camera_features['features'], testset_features['features'][:, query_index]
testset_feature_x


# In[58]:


# %%timeit

# get the similar camera feature from feature database
flann = pyflann.FLANN()
result, _ = flann.nn(database_features, testset_feature_x, 1, algorithm="kdtree", trees=8, checks=64)
result


# In[59]:


retrieved_camera_param = database_camera_features['cameras'][result[0]]
retrieved_camera_param


# In[60]:


retrieved_image = SyntheticUtil.camera_to_edge_image(
    retrieved_camera_param, worldcup_2014_mat['points'], worldcup_2014_mat['line_segment_index'],
    im_h=720, im_w=1280, line_width=4)

_imshow(retrieved_image, figsize=(6, 3)), retrieved_camera_param, retrieved_image.shape


# In[61]:


_imshow(testset_features['edge_map'][..., query_index], figsize=(6, 3)), testset_features['features'][..., query_index]


# In[62]:


_print_dictkeys_shape(worldcup_2014_mat)


# In[63]:


_print_dictkeys_shape(testset_features)


# ##  Code Refacting

# ### Camera Class

# In[64]:


@dataclass
class CameraProp(object):
    fl: float
    cx: float
    cy: float
    skew: float = 0.
    aspect: float = 1.


# This is **Extrinsic Pose**, not the camera pose with respect to the world.
class CameraPose(object):
    '''
    This is the camera extrinsic pose. Here is not the linear transform matrix (Camera Pose Transform)
    https://ksimek.github.io/2012/08/22/extrinsic/
    TODO the class name is not good. so easy to confuse the transform and extrinsic
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
        return CameraPose(rmat, offset).I  # Must

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
        self.P = self.K @ self.E

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
        return f'K:\n{self.K}\nE:\n{self.E}'


# #### Test CameraPose VS RotationUtil

# In[65]:


fl = 3081.976880
cx, cy = 640, 360
offset = np.asarray([52.816224, -54.753716, 19.960425])
rotx, roty, rotz = -90, 0.2, 0
pany, tiltx = 25, -10
point3d = np.asarray([[10, 20, 0]])


# In[66]:


prop = CameraProp(fl, cx, cy) 
base_pose = CameraPose.from_axis_xyz([rotx, roty, rotz], order='xyz', offset=offset)
pan_tilt_pose = CameraPose.from_axis_xyz([pany, tiltx], order='yx', offset=[0, 0, 0])
pose = pan_tilt_pose @ base_pose


# In[67]:


base_pose.r, pose.r, cv2.Rodrigues(pose.r)[0]


# In[68]:


base_rotation = RotationUtil.rotate_z_axis(rotz) @ RotationUtil.rotate_y_axis(roty) @ RotationUtil.rotate_x_axis(rotx)
pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pany, tiltx)
rotation = pan_tilt_rotation @ base_rotation
rot_vec, _ = cv2.Rodrigues(rotation)
base_rotation, rotation, rot_vec


# #### Test PerspectiveCamera VS ProjectiveCamera

# In[69]:


camera = PerspectiveCamera(prop, pose)
camera.P, camera.project_3d(point3d)


# In[70]:


camera2 = ProjectiveCamera(fl, cx, cy, offset, rot_vec)
camera2.P, camera2.project_3d(point3d[0][0], point3d[0][1], point3d[0][2], 1)


# #### Test Homegraphy
# 
# ![](http://assets.erlangai.cn/Misc/camera/homography.png)

# In[71]:


H = camera.get_homography()
H_inv = np.linalg.inv(H)
H, H.T, H_inv


# In[72]:


H2 = camera2.get_homography()
H2_inv = np.linalg.inv(H2)
H2, H2.T, H2_inv


# In[73]:


img16 = cv2.imread(f'{TOP_DIR}/data/UoT_soccer/16.jpg')
_imshow(img16, figsize=(12, 8))


# In[74]:


imgwraped = _warp_perspective(H, img16, (template_w, template_h), scale=10)
_imshow(imgwraped, figsize=(12, 8))


# ### Synthetic Class

# In[75]:


Stats = namedtuple('Stats', ['mean', 'std', 'min', 'max'], defaults=[0] * 4)
MotionStd = namedtuple('MotionStd', ['fl', 'pan', 'tilt'], defaults=[0] * 3)
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

    def generate_camera_pair(motion_param:CameraMotionParameter, motion_std:MotionStd, image_size:tuple, camera_num:int):
        rnd_normal = lambda stat, size: np.random.normal(stat.mean, stat.std, (camera_num, size))
        rnd_uniform = lambda stat: np.random.uniform(stat.min, stat.max, camera_num)
        rnd_nearby = lambda d, std: d + np.random.uniform(-0.5, 0.5, 1) * std

        ccs = rnd_normal(motion_param.cc, 3)
        fls = rnd_normal(motion_param.fl, 1)
        rolls = rnd_uniform(motion_param.roll)  # z-axis when in base axis, y-axis when in world axis
        pans = rnd_uniform(motion_param.pan)    # y-axis when in base axis
        tilts = rnd_uniform(motion_param.tilt)  # x-axis when in base axis

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
        """
        生成摄像机角度里的球场边线图像
        """
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


# #### PTZ Cameras Parameters

# In[76]:


wcp = worldcup_camera_parameter
num = 5 * 2

motion_param = CameraMotionParameter(
    cc = Stats(wcp['cc_mean'][0], wcp['cc_std'][0], wcp['cc_min'][0], wcp['cc_max'][0]),
    fl = Stats(wcp['fl_mean'][0], wcp['fl_std'][0], wcp['fl_min'][0], wcp['fl_max'][0]),
    pan = Stats(min=-35.0, max=35.0),
    roll = Stats(0, 0.2, -1.0, 1.0),
    tilt = Stats(min=-15.0, max=-5.0)
)

cameras = SyntheticDataset.generate_ptz_cameras(motion_param, (WIDTH, HEIGHT), num)


# In[77]:


soccer_field = SoccerField(worldcup_2014_mat['points'], worldcup_2014_mat['line_segment_index'])
image_edges = []
for i, camera_data in enumerate(cameras):
    image_edges.append(SyntheticDataset.generate_camera_image(camera_data, soccer_field, (WIDTH, HEIGHT), thickness=4))
_imshow(_imgrid(image_edges, nrow=5, padding=6), figsize=(18, 6))


# In[78]:


image_wraps = []
for camera_data, camera_image in zip(cameras, image_edges):
    prop = CameraProp(cx=camera_data[0], cy=camera_data[1], fl=camera_data[2])
    pose = CameraPose.from_axis_angle(camera_data[3:6], camera_data[6:9])
    camera = PerspectiveCamera(prop, pose)
    image_wraps.append(_warp_perspective(camera.get_homography(), camera_image, (template_w, template_h), scale=10))
_imshow(_imgrid(image_wraps, nrow=5, padding=6), figsize=(18, 6))


# #### Pivot and Positive Cameras

# In[79]:


motion_std = MotionStd(30, 1.5, 0.75)
cams_pivot, cams_positive = SyntheticDataset.generate_camera_pair(motion_param, motion_std, image_size=(WIDTH, HEIGHT), camera_num=num)
cams_pivot.shape, cams_positive.shape


# In[80]:


np.random.seed(123456)
indices = np.random.choice(num, min(5, num//2), False)
pivot_edges, positive_edges = [], []
for i in indices:
    pivot_edges.append(SyntheticDataset.generate_camera_image(cams_pivot[i], soccer_field, (WIDTH, HEIGHT), thickness=4))
    positive_edges.append(SyntheticDataset.generate_camera_image(cams_positive[i], soccer_field, (WIDTH, HEIGHT), thickness=4))


# In[81]:


meld_image = np.vstack((np.hstack(pivot_edges), np.hstack(positive_edges)))
_imshow(meld_image, figsize=(16, 8))


# #### Synthetic Siamese Network Dataset

# In[82]:


camera_parameter_file = f'{TOP_DIR}/data/worldcup_dataset_camera_parameter.mat'
soccer_field_template_file = f'{TOP_DIR}/data/worldcup2014.mat'
dataset_sample_file = f'{TOP_DIR}/data/dataset_sample.mat'
interpolation = cv2.INTER_AREA

def generate_siamese_dataset(camera_parameter_file, soccer_field_template_file, output_file = None, image_size=(320, 180), image_num=10000):
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
        image_piv = cv2.cvtColor(cv2.resize(image_piv, image_size, interpolation=interpolation), cv2.COLOR_BGR2GRAY)
        image_pos = cv2.cvtColor(cv2.resize(image_pos, image_size, interpolation=interpolation), cv2.COLOR_BGR2GRAY)
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


# In[83]:


dataset_sample_file = None
dataset_sample = generate_siamese_dataset(camera_parameter_file, soccer_field_template_file, dataset_sample_file, image_num=50)
if dataset_sample_file is not None:
    dataset_sample = sio.loadmat(dataset_sample_file)
_print_dictkeys_shape(dataset_sample)


# In[84]:


dataset_sample['image_mean'], dataset_sample['image_std']
# [[0.01840268]] [[0.11583157]]


# In[85]:


meld_image = np.vstack((np.hstack(dataset_sample['pivot_images'][:5]), np.hstack(dataset_sample['positive_images'][:5])))
_imshow(meld_image, figsize=(16, 8), canvas=False)


# ### Dataset Loader

# In[86]:


import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import make_grid 
from torch.utils.data import (Dataset, DataLoader)
from collections import namedtuple
from PIL import Image


CameraPair = namedtuple('CameraPair', ['anchor_images', 'positive_images'])

class CameraPairDataset(Dataset):

    def __init__(self, data:CameraPair, transform=None):
        self.data = data
        self.size = len(self.data.anchor_images)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((180, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0188], std=[0.128])
            ])
        else:
            self.transform = transform

        #
        # | 0, 1, 2, ..., N, 0, 1, 2, ..., N | --> x1
        # | 0, 1, 2, ..., N, roll(shift=1)   | --> x2
        # | 1, 1, 1, ..., 1, 0, 0, 0, ..., 0 | --> label
        #
        rng = np.random.default_rng(seed=123456)
        indices = rng.permutation(self.size)
        x1_inputs = np.concatenate((indices, indices), axis=0).reshape(-1, 1)
        x2_inputs = np.concatenate((indices, np.roll(indices, shift=1)), axis=0).reshape(-1, 1)
        self.indices = rng.permutation(np.concatenate((x1_inputs, x2_inputs), axis=1))

    def __len__(self):
        return 2 * self.size

    def __getitem__(self, index):
        ix1, ix2 = self.indices[index]
        x1 = self.data.anchor_images[ix1]
        x2 = self.data.anchor_images[ix2] if ix1 != ix2 else self.data.positive_images[ix2]
        return (self.transform(Image.fromarray(x1)),                 self.transform(Image.fromarray(x2)),                 torch.from_numpy(np.array([int(ix1 == ix2)], dtype=np.float32)))


# #### Test Iterator

# In[87]:


mean, std = 0.01852269, 0.11619838 # dataset_sample['image_mean'], dataset_sample['image_std']
data_transform = transforms.Compose([
    transforms.Resize((180, 320)), # H, W
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
data = CameraPair(anchor_images=dataset_sample['pivot_images'], positive_images=dataset_sample['positive_images'])
dataset = CameraPairDataset(data, transform=data_transform)
len(dataset), dataset_sample['pivot_camera'][0]


# In[88]:


nrow = 5
train_loader = DataLoader(dataset, batch_size=nrow, shuffle=True, num_workers=4)
for bx1, bx2, bla in train_loader:
    print(bx1.shape, bx2.shape, bla)
    break
bx1, bx2, bla = next(iter(train_loader))
bx1.shape, bx2.shape, bla


# In[89]:


grid_image = make_grid(torch.cat((bx1, bx2), dim=0), nrow=nrow, padding=3, normalize=True, pad_value=0.75)
grid_pil_image = transforms.ToPILImage()(grid_image)
grid_image.shape, grid_pil_image.convert('L').show()


# #### Convert cv2 / PIL / Tensor

# In[90]:


img36 = np.arange(36, dtype=np.uint8).reshape((6,6))
cv_img = img36 
type(cv_img), cv_img.shape, cv_img


# In[91]:


pil_img = Image.fromarray(cv_img)
type(pil_img), pil_img.width, pil_img.height


# In[92]:


tsr_img = torch.from_numpy(cv_img)
tsr_cv_img = transforms.ToTensor()(cv_img) # range: [0, 1]
tsr_pil_img = transforms.ToTensor()(pil_img)
tsr_img, tsr_cv_img.shape, tsr_cv_img, tsr_pil_img.shape, tsr_pil_img


# In[93]:


check_diff = tsr_cv_img == tsr_pil_img
torch.all(check_diff), np.all(check_diff.numpy()) 


# ---------
# 
# If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.

# In[94]:


# transforms only input Pil or tensor type
T_resize = transforms.Resize((3, 3), interpolation=transforms.InterpolationMode.BILINEAR)
T_tensor = transforms.ToTensor()
Image.BILINEAR, transforms.InterpolationMode.BILINEAR.value, cv2.INTER_LINEAR, cv2.INTER_AREA


# In[95]:


pil_resize_img = T_resize(pil_img)
tsr_pil_resize_img = T_tensor(pil_resize_img)
type(pil_resize_img), pil_resize_img.width, pil_resize_img.height, tsr_pil_resize_img.shape, tsr_pil_resize_img


# In[96]:


np.array(pil_resize_img), cv2.resize(cv_img, (3, 3), interpolation=cv2.INTER_AREA)


# -----------------

# In[97]:


tsr_chnl_img = tsr_img[None,::]
tsr_chnl_resize_img = T_resize(tsr_chnl_img)
tsr_tsr_resize_img = T_tensor(tsr_chnl_resize_img.numpy())
tsr_chnl_img, tsr_chnl_resize_img, tsr_tsr_resize_img


# ## Siamese Model

# ### Network 

# In[98]:


import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin:float):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        """
        similar:1, dissimilar:0
        """
        dist = F.pairwise_distance(x1, x2, keepdim=True)
        loss = torch.mean(
            label * torch.pow(dist, 2) + \
                (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss


class SiameseNetwork(nn.Module):
    def __init__(self):
        """
        inputs: (B, 1, 180, 320)

        filter_feature_size: W / ((K - 2 * Pad) * S), H / ((K - 2 * Pad) * S)
        """
        super(SiameseNetwork, self).__init__()
        self.extract_features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1, inplace=True),
            # (B, 4, 90, 160)
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            # (B, 8, 45, 80)
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # (B, 16, 23, 40)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # (B, 32, 12, 20)
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # (B, 16, 6, 10)
            # nn.Dropout(p=0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(960, 16), # 960: 6 * 10 * 16
            # nn.Linear(960, 480), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            # nn.Linear(480, 240), nn.ReLU(inplace=True), # nn.Dropout(p=0.5),
            # nn.Linear(240, 16)
        )

    def _forward_once(self, x):
        x = self.extract_features(x)
        x = self.fc(x.view(x.size(0), -1))
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        x1 = self._forward_once(x1)
        x2 = self._forward_once(x2)
        return x1, x2


# In[99]:


from torchsummary import summary

siamese = SiameseNetwork()
print(siamese)
summary(SiameseNetwork(), input_size=[(1, 180, 320), (1, 180, 320)], batch_size=3)


# ### Train

# In[100]:


import torch.optim as optim
import torch.backends.cudnn as cudnn

# model_save_path = f'{CKPTS_DIR}/siamese_nodropout_10000.pth'
# model_save_path = f'{CKPTS_DIR}/siamese_10000.pth'
model_save_path = f'{CKPTS_DIR}/siamese_15000_ori.pth'


# In[101]:


epoch_beg = 0
epoch_num = 5 
learning_rate = 0.01

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.cuda_id))
    cudnn.benchmark = True

model_state_dict, optim_state_dict = None, None
if os.path.exists(model_save_path):
    ckpts = torch.load(model_save_path, map_location=device)
    model_state_dict, optim_state_dict, epoch_beg = ckpts['model'], ckpts['optimizer'], ckpts['epoch']
epoch_beg, ckpts['epoch']


# In[102]:


criterion = ContrastiveLoss(margin=1.0).to(device)

siamese = SiameseNetwork().to(device)
if model_state_dict is not None:
    siamese.load_state_dict(model_state_dict)

# optimizer = optim.SGD(siamese.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(siamese.parameters(), lr=learning_rate)
if optim_state_dict is not None:
    optimizer.load_state_dict(optim_state_dict)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 55, 130], gamma=0.1)


# In[103]:


epoch = 0

for epoch in range(epoch_beg, epoch_beg + epoch_num + 1):
    siamese.train()
    losses, pos_dists, neg_dists = [], [], []
    for bx1, bx2, labels in train_loader:
        bx1, bx2, labels = bx1.to(device), bx2.to(device), labels.to(device)
        feat1, feat2 = siamese(bx1, bx2)
        loss = criterion(feat1, feat2, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        dist = F.pairwise_distance(feat1.detach(), feat2.detach(), keepdim=True)
        for l, d in zip(labels.detach().squeeze(), dist.squeeze()):
            pos_dists.append(d) if l == 1 else neg_dists.append(d)

    scheduler.step()

    dist_pos, dist_neg = torch.mean(torch.tensor(pos_dists)), torch.mean(torch.tensor(neg_dists))
    print('[%d] lr=[%.6f] loss[%.6f] pos_d[%.6f] neg_d[%.6f] ratio[%.6f]' % (
        epoch,
        optimizer.param_groups[0]['lr'],
        torch.mean(torch.tensor(losses)), dist_pos, dist_neg,
        dist_neg / (dist_pos + 0.000001) 
        ))

# torch.save({
#     'epoch': epoch,
#     'model': siamese.state_dict(),
#     'optimizer': optimizer.state_dict()
# }, model_save_path)
# epoch


# ### Test

# #### Batch Test

# In[104]:


test_sample = generate_siamese_dataset(camera_parameter_file, soccer_field_template_file, image_num=20)
_print_dictkeys_shape(test_sample)


# In[105]:


data = CameraPair(anchor_images=test_sample['pivot_images'], positive_images=test_sample['positive_images'])
test_dataset = CameraPairDataset(data, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)
len(test_dataset)


# In[106]:


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.cuda_id))
    cudnn.benchmark = True
    
model_state_dict = None
if os.path.exists(model_save_path):
    ckpts = torch.load(model_save_path, map_location=device)
    model_state_dict = ckpts['model']
else:
    raise
siamese = SiameseNetwork().to(device)
if model_state_dict is not None:
    siamese.load_state_dict(model_state_dict)


# In[107]:


siamese.eval()
with torch.no_grad():
    pos_dists, neg_dists = [], []
    for bx1, bx2, labels in test_loader:
        bx1, bx2, labels = bx1.to(device), bx2.to(device), labels.to(device)
        feat1, feat2 = siamese(bx1, bx2)
        dist = F.pairwise_distance(feat1.detach(), feat2.detach(), keepdim=True)
        print(np.hstack((dist.numpy(), labels.numpy())))


# In[108]:


test_loader_iter = iter(test_loader)
bx1, bx2, labels = next(test_loader_iter)
bx1, bx2, labels = bx1.to(device), bx2.to(device), labels.to(device)
feat1, feat2 = siamese(bx1, bx2)
dist = F.pairwise_distance(feat1.detach(), feat2.detach(), keepdim=True)
num_each_row = bx1.shape[0]
print(np.hstack((dist.numpy(), labels.numpy())))
grid_image = make_grid(torch.cat((bx1, bx2), dim=0), nrow=num_each_row, padding=3, normalize=True, pad_value=0.75)
grid_pil_image = transforms.ToPILImage()(grid_image)
grid_image.shape, grid_pil_image.convert('L').show()


# #### Single Test

# In[109]:


test_001_path = f'{TOP_DIR}/datasets/soccer_seg_detection/single/test_001.jpg'
test_001_det_path = f'{TOP_DIR}/datasets/soccer_seg_detection/det_test_001_fake.png'


# In[110]:


img_test_001, img_test_001_det = cv2.imread(test_001_path), cv2.imread(test_001_det_path)
img_test_001.shape, img_test_001_det.shape


# In[111]:


_imshow(np.hstack([cv2.resize(img_test_001, img_test_001_det.shape[:2]), img_test_001_det]))


# In[112]:


def extract_siamese_feature(x, net, transform=None):
    net.eval()
    if len(x.shape) == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((180, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.01848, std=0.11606)
        ])
    x = torch.unsqueeze(transform(Image.fromarray(x)), 0)
    with torch.no_grad():
        x = net._forward_once(x)
        x = x.cpu().numpy().squeeze()
    return x
test_001_feature = extract_siamese_feature(img_test_001_det, siamese, data_transform)
test_001_feature


# ### Features Database

# #### Generator Features

# In[113]:


class FieldEdgeImages(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.size = len(data)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((180, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0188], std=[0.128])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.data[index]))
        

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
    cameras = SyntheticDataset.generate_ptz_cameras(motion_param, image_size=(1280, 720), camera_num=image_num)
    
    image_list = []
    for i in range(image_num):
        image = SyntheticDataset.generate_camera_image(cameras[i], soccer_field, (1280, 720), thickness=4)
        image = cv2.cvtColor(cv2.resize(image, image_size, interpolation=interpolation), cv2.COLOR_BGR2GRAY)
        image_list.append(image)
        
    return {'images': np.asarray(image_list), 'cameras': cameras}


# In[114]:


num = 20000
feature_database_path = f'{TOP_DIR}/data/features_database_{num}_ori.mat'
if os.path.exists(feature_database_path):
    features_data  = sio.loadmat(feature_database_path)
else:
    features_data = generate_features_database(camera_parameter_file, soccer_field_template_file, image_num=num)
    dataset = FieldEdgeImages(features_data['images'], data_transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)

    ckpts = torch.load(model_save_path, map_location='cpu')
    siamese = SiameseNetwork()
    siamese.load_state_dict(ckpts['model'])
    siamese.eval()

    features = []
    with torch.no_grad():
        for x in data_loader:
            f = siamese._forward_once(x)
            features.append(f.cpu().numpy())
    features_data['features'] = np.vstack(features)
    sio.savemat(f'{TOP_DIR}/data/features_database_{num}_ori.mat', features_data)
features_data.keys()


# #### Find ANN by Flann

# In[115]:


TestImage16 = namedtuple('TestImage16', ['raw', 'edge', 'seg', 'det'])
imgs_list = [cv2.imread(f'{TOP_DIR}/data/UoT_soccer/{x}') for x in ('16.jpg', '16_edge_image.jpg', '016_AB_phase1.jpg',  '016_AB_phase2.jpg')]
test16 = TestImage16(imgs_list[0], imgs_list[1], imgs_list[2], imgs_list[3])
for im in imgs_list:
    print(im.shape)


# In[116]:


_imshow(np.hstack((test16.raw, test16.edge)), figsize=(12, 4))


# In[117]:


raw, edge = cv2.resize(test16.raw, (300, 300), interpolation=interpolation), cv2.resize(test16.edge, (300, 300), interpolation=interpolation)
_imshow(np.hstack((raw, edge)), figsize=(10, 10))


# In[118]:


_imshow(np.hstack((test16.seg, test16.det)), figsize=(12, 4))


# In[119]:


test16_det_edge = test16.det[:, 300:, :]
# test16_det_edge[test16_det_edge > 20] = 255
# test16_det_edge = cv2.threshold(test16_det_edge, 20, 255, cv2.THRESH_BINARY)[1]
edge1 = cv2.resize(test16.edge, (320, 180), interpolation=interpolation)
edge2 = cv2.resize(test16_det_edge, (320, 180), interpolation=interpolation)
_imshow(np.abs(edge2 - edge1), figsize=(12, 5))


# In[120]:


f1 = extract_siamese_feature(edge1, siamese, data_transform)
f2 = extract_siamese_feature(edge2, siamese, data_transform)
[f'{x:10.5} {y:10.5}' for x, y in zip(f1, f2)]


# In[121]:


pyflann.set_distance_type(distance_type='euclidean')
flann = pyflann.FLANN()
params = flann.build_index(features_data['features'], algorithm='kdtree', trees=8, checks=64)
nbrs1, dists1 = flann.nn_index(f1, 5)
nbrs2, dists2 = flann.nn_index(f2, 5)
params.keys(), params['iterations'], nbrs1, dists1, nbrs2, dists2


# In[122]:


[f'{a:10.5} {b:10.5}, {c:10.5}, {d:10.5}' for a, b, c, d in zip(f1, f2, features_data['features'][nbrs1[0][0]], features_data['features'][nbrs2[0][0]])]


# In[123]:


images_f1_nbrs = features_data['images'][nbrs1[0]]
images_f2_nbrs = features_data['images'][nbrs2[0]]
images_bchw_f1_nbrs = np.expand_dims(images_f1_nbrs, axis=1)
images_bchw_f2_nbrs = np.expand_dims(images_f2_nbrs, axis=1)

images_bchw_f1_nbrs.shape, images_bchw_f2_nbrs.shape


# In[124]:


_imshow(_imgrid(images_f1_nbrs), figsize=(18, 4))


# In[125]:


_imshow(_imgrid(images_f2_nbrs), figsize=(18, 4))


# ## Homography and Refine

# ### Before

# In[126]:


retrieved_camera_data16 = features_data['cameras'][nbrs1[0][0]]
retrieved_camera_data16


# In[127]:


prop = CameraProp(cx=retrieved_camera_data16[0], cy=retrieved_camera_data16[1], fl=retrieved_camera_data16[2])
pose = CameraPose.from_axis_angle(retrieved_camera_data16[3:6], retrieved_camera_data16[6:9])
camera16 = PerspectiveCamera(prop, pose)
homo16 = camera16.get_homography()
himg_before = _warp_perspective(homo16, test16.raw, (template_w, template_h), scale=10)
homo16, _imshow(_imgrid((test16.raw, himg_before)), figsize=(22, 6))


# ### After

# In[128]:


retrieved_img16 = SyntheticDataset.generate_camera_image(retrieved_camera_data16, soccer_field, image_size=(1280, 720))
retrieved_img16_db = features_data['images'][nbrs1[0][0]]
_imshow(_imgrid([test16.edge, retrieved_img16, retrieved_img16_db]), color='gray', figsize=(22, 6))


# In[129]:


retrieved_img16_bin = cv2.threshold(cv2.cvtColor(retrieved_img16, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY_INV)[1]
test_edge_img16_bin = cv2.threshold(cv2.cvtColor(test16.edge, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY_INV)[1]
_imshow(_imgrid([retrieved_img16_bin, test_edge_img16_bin], pad_value=0), color='gray', figsize=(16, 6))


# In[130]:


retrieved_img16_dist = cv2.distanceTransform(retrieved_img16_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
test_edge_img16_dist = cv2.distanceTransform(test_edge_img16_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
_imshow(_imgrid([retrieved_img16_dist, test_edge_img16_dist], pad_value=30), color='gray', figsize=(16, 6))


# In[131]:


dist_threshold = 50
retrieved_img16_dist[retrieved_img16_dist > dist_threshold] = dist_threshold
test_edge_img16_dist[test_edge_img16_dist > dist_threshold] = dist_threshold
_imshow(_imgrid([retrieved_img16_dist, test_edge_img16_dist], pad_value=255), color='gray', figsize=(16, 6))


# In[132]:


warp = np.eye(3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
_, warp = cv2.findTransformECC(retrieved_img16_dist, test_edge_img16_dist, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)


# In[133]:


hrefine = warp @ homo16
hrefine


# In[134]:


himg_after = _warp_perspective(hrefine, test16.raw, (template_w, template_h), scale=10)
homo16, _imshow(himg_after, figsize=(12, 6))


# In[135]:


_imshow(_imgrid([test16.raw, test16.edge, himg_before, himg_after], nrow=2, padding=6), figsize=(16,10))


# ### Together

# In[136]:


# count: 186
test_root_path = f'{TOP_DIR}/datasets/soccer_seg_detection'
test_rawtst_path = f'{test_root_path}/raw/test'
test_output_path = f'{test_root_path}/output/test'
test_result_path = f'{test_root_path}/output/result'

test_images_list = []
for item in os.scandir(f'{test_root_path}/raw/test'):
    if item.name[-3:] != 'jpg':
        continue
    test_images_list.append(item.name)


# In[137]:


TestImage = namedtuple('TestImage', ['raw', 'x', 'seg_mask', 'det_mask', 'det_edge', 'gt_edge'])
ResultImage = namedtuple('ResultImage', ['homo_refine', 'homo_origin', 'search_dist', 'detect_dist'])

def create_test_sample(img_name):
    idx = int(img_name.split('.')[0])
    return TestImage(
        raw=cv2.imread(f'{test_rawtst_path}/{img_name}'),
        x=cv2.imread(f'{test_output_path}/seg_{idx}_real.png'),
        seg_mask=cv2.imread(f'{test_output_path}/seg_{idx}_fake.png', cv2.IMREAD_GRAYSCALE),
        det_mask=cv2.imread(f'{test_output_path}/det_{idx}_real.png'),
        det_edge=cv2.imread(f'{test_output_path}/det_{idx}_fake.png', cv2.IMREAD_GRAYSCALE),
        gt_edge=cv2.imread(f'{test_root_path}/test/{idx:03}_AB.jpg', cv2.IMREAD_GRAYSCALE)[:, 256:]
    )

def birdeye_perspective_transform(network, flann, features_db, img_raw, img_edge, debug=False):
    result = {'images':{}}
    template_w, template_h = 107, 69
    if len(img_edge.shape) == 3:
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)
    
    def _extract_siamese_feature(x, nete):
        net.eval()
        transform = transforms.Compose([
            transforms.Resize((180, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0185, std=0.1161)
        ])
        x = torch.unsqueeze(transform(Image.fromarray(x)), 0)
        with torch.no_grad():
            x = net._forward_once(x)
            x = x.cpu().numpy().squeeze()
        return x
    
    def _warp_perspective(h_templ2cam, img_cam, img_size, scale=1, bg=0):
        # template view (0, 0) at top-left, so v-flip transform 
        trans = np.array([
            [1, 0, 0],
            [0, -1, img_size[1]],
            [0, 0, 1],
        ])
        if scale > 1:
            trans = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ]) @ trans
        h_cam2templ = np.linalg.inv(h_templ2cam)
        return cv2.warpPerspective(img_cam, trans @ h_cam2templ, (scale * img_size[0], scale * img_size[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    
    img_edge = cv2.resize(img_edge, (320, 180), interpolation=cv2.INTER_AREA)
    features = extract_siamese_feature(img_edge, network)
    nbrs, dists = flann.nn_index(features, 5 if debug else 1)
    if debug:
        nbrs, dists = nbrs[0], dists[0]
    camera_data = features_db['cameras'][nbrs[0]]
    prop = CameraProp(cx=camera_data[0], cy=camera_data[1], fl=camera_data[2])
    pose = CameraPose.from_axis_angle(camera_data[3:6], camera_data[6:9])
    camera = PerspectiveCamera(prop, pose)
    
    img_edge_src = cv2.resize(features_db['images'][nbrs[0]], (1280, 720))
    img_edge_dst = cv2.resize(img_edge, (1280, 720))
    img_edge_search = cv2.threshold(img_edge_src, 10, 255, cv2.THRESH_BINARY_INV)[1]
    img_edge_detect = cv2.threshold(img_edge_dst, 10, 255, cv2.THRESH_BINARY_INV)[1]
    
    img_search_dist = cv2.distanceTransform(img_edge_search, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    img_detect_dist = cv2.distanceTransform(img_edge_detect, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_threshold = 50
    img_search_dist[img_search_dist > dist_threshold] = dist_threshold
    img_detect_dist[img_detect_dist > dist_threshold] = dist_threshold
    
    homo = camera.get_homography()
    if debug:
        result['images']['dist: %.6f' % dists[0]] = img_edge_src
        result['images']['search_dist'] = cv2.normalize(img_search_dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        result['images']['detect_dist'] = cv2.normalize(img_detect_dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        result['images']['homo_origin'] = _warp_perspective(homo, img_raw, (template_w, template_h), scale=10)
        
    hrefine = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    try:
        hrefine = cv2.findTransformECC(img_search_dist, img_detect_dist, hrefine, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)[1]
    except Exception:
        pass
    result['images']['homo_refine'] = _warp_perspective(hrefine @ homo, img_raw, (template_w, template_h), scale=10)
    result['ann_dist'] = dists[0]
    return result


# In[138]:


timg_nt = create_test_sample(test_images_list[66])
_imshow(_imgrid(
    [timg_nt.x, timg_nt.seg_mask, timg_nt.det_mask, timg_nt.det_edge, timg_nt.gt_edge],
    labels=['x', 'seg mask', 'det mask', 'det edge', 'gt edge'], font_scale=1.0
), figsize=(18, 4))


# In[139]:


result = birdeye_perspective_transform(siamese, flann, features_data, timg_nt.raw, timg_nt.det_edge, debug=True)
_imshow(_imgrid(result['images'], nrow=3, font_scale=3.0, text_color=(0,), text_color_bg=(255, 255, 255)), figsize='auto')


# In[140]:


if not os.path.exists(test_result_path):
    os.makedirs(test_result_path)
    
for iname in test_images_list:
    timg_nt = create_test_sample(iname)
    img_p1 = _imgrid(
        [timg_nt.x, timg_nt.seg_mask, timg_nt.det_mask, timg_nt.det_edge, timg_nt.gt_edge],
        labels=['x', 'seg mask', 'det mask', 'det edge', 'gt edge'], font_scale=1.0
    )
    result = birdeye_perspective_transform(siamese, flann, features_data, timg_nt.raw, timg_nt.det_edge, debug=True)
    img_p2 = _imgrid(result['images'], nrow=3, font_scale=3.0, text_color=(0,), text_color_bg=(255, 255, 255))
    (h1, w1), (h2, w2) = img_p1.shape[:2], img_p2.shape[:2]
    tsize1, tsize2 = (2048, int(2048 * h1 / w1)), (2048, int(2048 * h2 / w2))
    img1, img2 = cv2.resize(img_p1, tsize1), cv2.resize(img_p2, tsize2)
    cv2.imwrite(f'{test_result_path}/{iname}', np.vstack([img1, img2]))


# In[143]:


_imshow(cv2.imread(f'{test_result_path}/67.jpg'), figsize='auto')


# ## Referencs

# - https://wikiless.org/wiki/Rotation_matrix?lang=en
# - https://nhoma.github.io/ "数据集"
# - https://github.com/lood339/SCCvSD/
# - https://builtin.com/machine-learning/siamese-network
# - [Dissecting the Camera Matrix, Part 2: The Extrinsic Matrix][1]
# - [Estimating a Homography Matrix][2]
# 
# [2]: https://towardsdatascience.com/estimating-a-homography-matrix-522c70ec4b2c
# [1]: https://ksimek.github.io/2012/08/22/extrinsic/
