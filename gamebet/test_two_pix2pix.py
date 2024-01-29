#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file test_two_pix2pix.py
# @brief
# @author QRS
# @version 1.0
# @date 2023-10-27 17:27


import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


def save_images(visuals, prefix, image_path, img_dir):
    image_name = os.path.basename(image_path).split('.')[0]
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        img_name = '%s_%s_%s.png' % (prefix, image_name, label)
        save_path = os.path.join(img_dir, img_name)
        util.save_image(im, save_path, aspect_ratio=1.0)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # opt.name = 'soccer_seg_pix2pix'
    opt.name = ''
    opt.epoch = 'seg'
    model_seg = create_model(opt)      # create a model given opt.model and other options
    model_seg.setup(opt)

    # opt.name = 'soccer_det_pix2pix'
    opt.name = ''
    opt.epoch = 'det'
    model_det = create_model(opt)
    model_det.setup(opt)

    for i, data in enumerate(dataset):
        model_seg.set_input(data)
        print(model_seg.real.shape)
        model_seg.test()
        visuals = model_seg.get_current_visuals()
        img_path = model_seg.get_image_paths()
        save_images(visuals, 'seg', img_path[0], opt.checkpoints_dir)

        fake = (visuals['fake'] + 1.0) / 2.0
        real = (visuals['real'] + 1.0) / 2.0
        data['A'] = (fake * real) * 2.0 - 1

        model_det.set_input(data)
        model_det.test()
        visuals = model_det.get_current_visuals()
        img_path = model_det.get_image_paths()
        save_images(visuals, 'det', img_path[0], opt.checkpoints_dir)
