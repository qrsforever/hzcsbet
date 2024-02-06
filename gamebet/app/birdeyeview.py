#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import aux.constant as C
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from aux.camera import (
    CameraProp,
    CameraPose,
    PerspectiveCamera,
)


class BEVBaseExecutor(ExecutorBase):
    _name = 'BEVView'
    _feature_db = C.FEATURES_CAMERAS_HOG_PARAMS_PATH

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        frame, view = shared_frames[0], shared_frames[1]
        camera_data, dist = cache['find_params'](msg.pitch_feats)  # pyright: ignore
        prop = CameraProp(cx=camera_data[0], cy=camera_data[1], fl=camera_data[2])
        pose = CameraPose.from_axis_angle(camera_data[3:6], camera_data[6:9])
        camera = PerspectiveCamera(prop, pose)
        homo = camera.get_homography()
        # template to camera
        homo = cache['refine_camera_pose'](camera, view) @ homo
        # camera to template
        homo = np.linalg.inv(homo)
        # template view (0, 0) at top-left, so v-flip transform
        vflip_mat = np.array([
            [1, 0, 0],
            [0, -1, C.TEMPLATE_H],
            [0, 0, 1],
        ])
        homo = homo @ vflip_mat
        msg.homography_matrix = homo.tolist()
        return msg

    def pre_loop(self, cache):
        import scipy.io as sio
        import pyflann

        worldcup_2014_mat = sio.loadmat(C.WORLDCUP2014_TEMPL_PATH)
        points = worldcup_2014_mat['points']
        segmts = worldcup_2014_mat['line_segment_index']
        pts_3d = np.hstack((points, np.ones((points.shape[0], 1))))


        pyflann.set_distance_type(distance_type='euclidean')
        flann = pyflann.FLANN()
        features_cameras_params = sio.loadmat(self._feature_db_path)
        flann.build_index(features_cameras_params['features'], algorithm='kdtree', trees=8, checks=64)

        dtype = features_cameras_params['features'][0].dtype

        def find_camera_params(feats):
            nbrs, dists = flann.nn_index(np.asarray(feats, dtype=dtype), 1)
            return features_cameras_params['cameras'][nbrs[0]], dists[0]

        def project_camera_court(camera, image_size):
            image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            pts_2d = camera.project_3d(pts_3d)
            for idx1, idx2 in segmts:
                p1, p2 = np.rint(pts_2d[idx1]).astype(np.int32), np.rint(pts_2d[idx2]).astype(np.int32)
                cv2.line(image, tuple(p1), tuple(p2), (255, 255, 255), thickness=4)
            return image

        def refine_camera_pose(camera, detededge_image):
            retrieved_image = np.zeros((detededge_image.shape[1], detededge_image.shape[0], 3), dtype=np.uint8)
            pts_2d = camera.project_3d(pts_3d)
            for idx1, idx2 in segmts:
                p1, p2 = np.rint(pts_2d[idx1]).astype(np.int32), np.rint(pts_2d[idx2]).astype(np.int32)
                cv2.line(retrieved_image, tuple(p1), tuple(p2), (255, 255, 255), thickness=4)

            def distance_transform(image):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
                return cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            detededge_dist = distance_transform(detededge_image)
            retrieved_dist = distance_transform(retrieved_image)

            dist_threshold = 50
            detededge_dist[detededge_dist > dist_threshold] = dist_threshold
            retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

            h_retrieved_to_edge = np.eye(3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
            try:
                _, h_retrieved_to_edge = cv2.findTransformECC(
                    retrieved_dist, detededge_dist, h_retrieved_to_edge,
                    cv2.MOTION_HOMOGRAPHY, criteria, None, 5) # type: ignore
            except Exception as err:
                self.logger.error(f'findTransformECC error: {err}')
            return h_retrieved_to_edge

        cache['find_params'] = find_camera_params
        cache['project_court'] = project_camera_court
        cache['refine_camera_pose'] = refine_camera_pose


class BEVSiameseExecutor(BEVBaseExecutor):
    _name = 'BEVSiamese'
    _feature_db_path = C.FEATURES_CAMERAS_PARAMS_PATH
    _model_weights_path = C.SIAMESE_WEIGHTS_PATH

    def __init__(self, use_gpu):
        super().__init__()
        self._use_gpu = use_gpu

    def siamese_infer(self, weights_path):
        from models.siamese import SiameseNetwork

        device = torch.device('cuda:0') if self._use_gpu else torch.device('cpu')

        # width: 320 height: 180
        transform = torch.nn.Sequential(
            T.Resize((180, 320), T.InterpolationMode.BICUBIC, antialias=False),  # type: ignore
            T.Normalize(mean=0.01848, std=0.11606)
        ).to(device)

        netS = SiameseNetwork()
        netS.load_state_dict(torch.load(weights_path, map_location=device)['model'])
        netS.to(device)
        netS.eval()

        @torch.inference_mode()
        def model_infer(inputs, trans=True):
            if len(inputs.shape) == 3:
                inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2GRAY)
            inputs = F.to_tensor(inputs)
            inputs = transform(inputs) if trans else inputs
            x = torch.unsqueeze(inputs, dim=0).to(device)
            x = netS._forward_once(x)
            return inputs.cpu().numpy(), x.cpu().numpy().squeeze()
        return model_infer

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        _, sme_out = cache['sme_infer'](shared_frames[1])
        msg.pitch_feats = sme_out.tolist()
        super().run(shared_frames, msg, cache)
        return msg

    def pre_loop(self, cache):
        super().pre_loop(cache)
        cache['sme_infer'] = self.siamese_infer(self._model_weights_path)


class BEVHogExecutor(BEVBaseExecutor):
    _name = 'BEVHog'
    _feature_db_path = C.FEATURES_CAMERAS_HOG_PARAMS_PATH

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        msg.pitch_feats = cache['compute_feature'](shared_frames[1])
        super().run(shared_frames, msg, cache)
        return msg

    def pre_loop(self, cache):
        super().pre_loop(cache)

        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (32, 32)
        cell_size = (32, 32)
        n_bins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
        def compute_feature(edge_image):
            edge_image = cv2.resize(edge_image, win_size)
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_RGB2GRAY)
            feat = hog.compute(edge_image)
            return feat.tolist()

        cache['compute_feature'] = compute_feature


class BEVRobustSFRExecutor(ExecutorBase):
    _name = "BEVSFR"
    _model_weights_path = C.ROUBUST_SFR_WEIGHTS_PATH

    def __init__(self, use_gpu):
        super().__init__()
        self._use_gpu = use_gpu

    def sfr_infer(self, weights_path):
        from models.robustsfr import EncDec

        device = torch.device('cuda:0') if self._use_gpu else torch.device('cpu')

        # width: 320 height: 180
        transform = torch.nn.Sequential(
            T.Resize((720, 1280), T.InterpolationMode.BICUBIC, antialias=False),  # type: ignore
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])  # ImageNet
        ).to(device)

        netS = EncDec(18, 92, 1)
        netS.load_state_dict(torch.load(weights_path, map_location=device)['model_state_dict'])
        netS.to(device)
        netS.eval()

        def cal_homography(scores, preds, tmpl_grid, nms_thres=0.995, num_classes=92):
            src_pts, dst_pts = [], []
            for cls in range(1, num_classes):
                pred_inds = preds == cls
                if not np.any(pred_inds):
                    continue
                values = scores[pred_inds]
                max_score = values.max()
                max_index = values.argmax()

                indices = np.where(pred_inds)
                coords = list(zip(indices[0], indices[1]))
                if max_score >= nms_thres:
                    x, y = coords[max_index]
                    src_pts.append((y * 4, x * 4)) # 320, 180 -> 1280, 720
                    dst_pts.append(tmpl_grid[cls])

            src_pts, dst_pts = np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)
            if len(src_pts) > 4:
                pred_homo, _ = cv2.findHomography(
                    src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10)
            else:
                pred_homo = np.array([])
            return pred_homo

        @torch.inference_mode()
        def model_infer(inputs, tmpl_grid):
            inputs = F.to_tensor(inputs)
            inputs = transform(inputs)
            x = torch.unsqueeze(inputs, dim=0).to(device)
            x = netS(x)  # [1, 92, 180, 320]
            x = torch.softmax(x, dim=1)  # [1, 92, 180, 320] （normalize: 0 ~ 1)
            values, indices = torch.max(x, dim=1)
            scores, preds = values[0].cpu().numpy(), indices[0].cpu().numpy()
            homo = cal_homography(scores, preds, tmpl_grid)
            return homo
        return model_infer

    def pre_loop(self, cache):
        def gen_template_grid():
            field_dim_x, field_dim_y = 114.83, 74.37  # in yard
            nx, ny = (13, 7)  # 91
            x = np.linspace(0, field_dim_x, nx)
            y = np.linspace(0, field_dim_y, ny)
            xv, yv = np.meshgrid(x, y, indexing='ij')
            uniform_grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
            template_cls_dict = {}
            for idx, pts in enumerate(uniform_grid):
                template_cls_dict[idx + 1] = pts
            return template_cls_dict

        cache['tmpl_grid'] = gen_template_grid()
        cache['sfr_infer'] = self.sfr_infer(self._model_weights_path)

    def run(self, shared_frames: tuple[np.ndarray, ...], msg: SharedResult, cache: dict) -> SharedResult:
        homo = cache['sfr_infer'](shared_frames[0], cache['tmpl_grid'])
        msg.homography_matrix = homo.tolist()
        return msg
