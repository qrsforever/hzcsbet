#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Pix2PixExecutor(ExecutorBase):

    def __init__(self, use_gpu):
        super().__init__()
        self._use_gpu = use_gpu

    def unet256_infer(self, weights_path):
        from models import networks

        device = torch.device('cuda:0') if self._use_gpu else torch.device('cpu')

        transform = torch.nn.Sequential(
            T.Resize((256, 256), T.InterpolationMode.BICUBIC, antialias=False),  # type: ignore
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ).to(device)

        netG = networks.UnetGenerator(3, 1, 8, 64, use_dropout=False)
        netG.load_state_dict(torch.load(weights_path, map_location=device))
        netG.to(device)
        netG.eval()

        @torch.inference_mode()
        def model_infer(inputs, trans=True):
            inputs = F.to_tensor(inputs)
            inputs = transform(inputs) if trans else inputs
            x = torch.unsqueeze(inputs, dim=0).to(device)
            x = netG(x)
            return inputs.cpu().numpy(), x.cpu().numpy().squeeze()
        return model_infer

    def siamese_infer(self, weights_path):
        from models import siamese

        device = torch.device('cuda:0') if self._use_gpu else torch.device('cpu')

        # width: 320 height: 180
        transform = torch.nn.Sequential(
            T.Resize((180, 320), T.InterpolationMode.BICUBIC, antialias=False),  # type: ignore
            T.Normalize(mean=0.01848, std=0.11606)
        ).to(device)

        netS = siamese.SiameseNetwork()
        netS.load_state_dict(torch.load(weights_path, map_location=device)['model'])
        netS.to(device)
        netS.eval()

        @torch.inference_mode()
        def model_infer(inputs, trans=True):
            if len(inputs.shape) == 3:
                inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2GRAY)
            inputs = F.to_tensor(inputs)
            inputs = transform(inputs) if trans else inputs
            x = torch.unsqueeze(inputs, dim=0).to(device)
            x = netS._forward_once(x)
            return inputs.cpu().numpy(), x.cpu().numpy().squeeze()
        return model_infer


class Pix2PixSegExecutor(Pix2PixExecutor):
    _name = "Pix2Seg"

    def __init__(self, weights_path, use_gpu):
        super().__init__(use_gpu)
        self._seg_weights_path = weights_path

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        # seg_out shape: (256 256)
        _, seg_out = cache['seg_infer'](cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgmsk_np = np.asarray((seg_out + 1) / 2.0 * 255, dtype=np.uint8)
        imgmsk_np = cv2.resize(imgmsk_np, frame.shape[:2][::-1])
        frame[:] = cv2.bitwise_and(frame, frame, mask=imgmsk_np)
        return msg

    def pre_loop(self, cache):
        cache['seg_infer'] = self.unet256_infer(self._seg_weights_path)


class Pix2PixDetExecutor(Pix2PixExecutor):
    _name = "Pix2Det"

    def __init__(self, weights_path, use_gpu):
        super().__init__(use_gpu)
        self._det_weights_path = weights_path

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        # det_out shape: (256 256)
        _, det_out = cache['det_infer'](cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgrgb_np = np.tile(det_out, (3, 1, 1)) # gray to rgb
        imgrgb_np = (np.transpose(imgrgb_np, (1, 2, 0)) + 1) / 2.0 * 255.0
        imgrgb_np = imgrgb_np.astype(dtype=np.uint8)
        edges = cv2.resize(imgrgb_np, frame.shape[:2][::-1])
        frame[:] = cv2.addWeighted(frame, 0.6, edges, 0.4, 0)
        return msg

    def pre_loop(self, cache):
        cache['det_infer'] = self.unet256_infer(self._det_weights_path)


class Pix2PixDetSiameseExecutor(Pix2PixExecutor):
    _name = "Pix2DetSiamese"

    def __init__(self, det_weights_path, sme_weights_path, use_gpu):
        super().__init__(use_gpu)
        self._det_weights_path = det_weights_path
        self._sme_weights_path = sme_weights_path

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        _, det_out = cache['det_infer'](cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det_out = np.asarray((det_out + 1) / 2.0 * 255, dtype=np.uint8)
        _, sme_out = cache['sme_infer'](det_out)
        msg.pitch_feats = sme_out.tolist()
        return msg

    def pre_loop(self, cache):
        cache['det_infer'] = self.unet256_infer(self._det_weights_path)
        cache['sme_infer'] = self.siamese_infer(self._sme_weights_path)


class Pix2PixSegDetSiameseExecutor(Pix2PixExecutor):
    _name = "Pix2SegDetSiamese"

    def __init__(self, seg_weights_path, det_weights_path, sme_weights_path,
                 use_gpu, blend_seg=True, blend_det=False):
        super().__init__(use_gpu)
        self._seg_weights_path = seg_weights_path
        self._det_weights_path = det_weights_path
        self._sme_weights_path = sme_weights_path
        self._blend_seg = blend_seg
        self._blend_det = blend_det

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        # seg_in shape: (1920, 1080, 3) seg_out shape: (256, 256)
        seg_in, seg_out = cache['seg_infer'](cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # seg_real shape: (3, 256, 256) seg_fake shape: (256, 256)
        seg_real, seg_fake = (seg_in + 1.0) / 2.0, (seg_out + 1.0) / 2.0
        if self._blend_seg:
            msk = np.asarray(seg_fake * 255, dtype=np.uint8)
            msk = cv2.resize(msk, frame.shape[:2][::-1])
            frame[:] = cv2.bitwise_and(frame, frame, mask=msk)

        # det_real shape: (3, 256, 256)
        det_real = np.transpose(((seg_real * seg_fake) * 2.0 - 1), (1, 2, 0))
        # det_out shape: (256 256)
        _, det_out = cache['det_infer'](det_real, trans=False)
        det_out = np.asarray((det_out + 1) / 2.0 * 255, dtype=np.uint8)
        if self._blend_det:
            # (256, 256, 3)
            rgb = np.transpose(np.tile(det_out, (3, 1, 1)), (1, 2, 0))
            rgb = cv2.resize(rgb, frame.shape[:2][::-1])
            frame[:] = cv2.addWeighted(frame, 0.6, rgb, 0.4, 0)

        _, sme_out = cache['sme_infer'](det_out)
        msg.pitch_feats = sme_out.tolist()
        return msg

    def pre_loop(self, cache):
        cache['seg_infer'] = self.unet256_infer(self._seg_weights_path)
        cache['det_infer'] = self.unet256_infer(self._det_weights_path)
        cache['sme_infer'] = self.siamese_infer(self._sme_weights_path)
