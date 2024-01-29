#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import numpy as np


class Pix2PixExecutor(ExecutorBase):

    def __init__(self, weights_path, use_gpu=False):
        super().__init__()
        self._weights_path = weights_path
        self._use_gpu = use_gpu

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        frame[:] = cache['pix2pix_infer'](frame)
        return msg

    def pre_loop(self, cache):
        import cv2
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
            mask = fake.cpu().numpy().squeeze()
            mask = np.asarray((mask + 1) / 2.0 * 255, dtype=np.uint8)
            mask = cv2.resize(mask, frame.shape[:2][::-1])
            return cv2.bitwise_and(frame, frame, mask=mask)

        cache['pix2pix_infer'] = model_infer

    def post_loop(self, cache):
        pass


class Pix2PixSegExecutor(Pix2PixExecutor):
    _name = "Pix2PixSeg"

class Pix2PixDetExecutor(Pix2PixExecutor):
    _name = "Pix2PixDet"
