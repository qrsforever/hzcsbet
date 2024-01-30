#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import numpy as np
import cv2


class Pix2PixExecutor(ExecutorBase):

    def __init__(self, weights_path, use_gpu=False):
        super().__init__()
        self._weights_path = weights_path
        self._use_gpu = use_gpu

    def pre_loop(self, cache):
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from models import networks

        def transform(frame):
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return T.Compose([
                T.Resize((256, 256), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(image_pil)

        device = torch.device('cuda:0') if self._use_gpu else torch.device('cpu')
        netG = networks.UnetGenerator(3, 1, 8, 64, use_dropout=False)
        netG.load_state_dict(torch.load(self._weights_path, map_location=device))
        netG.to(device)
        netG.eval()

        @torch.inference_mode()
        def model_infer(frame):
            real = transform(frame)
            input = torch.unsqueeze(real, dim=0).to(device)
            fake = netG(input)
            return fake.cpu().numpy().squeeze()

        cache['infer'] = model_infer


class Pix2PixSegExecutor(Pix2PixExecutor):
    _name = "Pix2PixSeg"

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        image_gray_numpy = cache['infer'](frame) # shape: (256 256)
        image_mask_numpy = np.asarray((image_gray_numpy + 1) / 2.0 * 255, dtype=np.uint8)
        image_mask_numpy = cv2.resize(image_mask_numpy, frame.shape[:2][::-1])
        frame[:] = cv2.bitwise_and(frame, frame, mask=image_mask_numpy)
        return msg

class Pix2PixDetExecutor(Pix2PixExecutor):
    _name = "Pix2PixDet"

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        image_gray_numpy = cache['infer'](frame) # shape: (256 256)
        image_rgb_numpy = np.tile(image_gray_numpy, (3, 1, 1)) # gray to rgb
        image_rgb_numpy = (np.transpose(image_rgb_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_rgb_numpy = image_rgb_numpy.astype(dtype=np.uint8)
        edges = cv2.resize(image_rgb_numpy, frame.shape[:2][::-1])
        frame[:] = cv2.addWeighted(frame, 0.6, edges, 0.4, 0)
        return msg
