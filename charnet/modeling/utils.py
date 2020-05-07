# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import math


def rotate_rect(x1, y1, x2, y2, degree, center_x, center_y):
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    new_points = list()
    for point in points:
        dx = point[0] - center_x
        dy = point[1] - center_y
        new_x = center_x + dx * math.cos(degree) - dy * math.sin(degree)
        new_y = center_y + dx * math.sin(degree) + dy * math.cos(degree)
        new_points.append([(new_x), (new_y)])
    return new_points

def draw_bboxes(img, bboxes, color=(0, 0, 255), thickness=1):
    for bbox in bboxes:
        # if [x1, y1, x2, y2]
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[-2:]), color, thickness)
        # if [x1, y1, width, height]
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[:2]+bbox[-2:]), color, thickness)


# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:18 AM
# @Author  : zhoujun
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def show_img(imgs: np.ndarray, titles, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img, title in zip(imgs, titles):
        fig = plt.figure()
        fig.suptitle(title)
        plt.imshow(img, cmap=None if color else 'gray')

def to_img(x, mean=0, std=1, c=1, h=28,w=28):
    '''resize image to B,C,H,W after unnormalization. 
    Input : 
        x: of size B, _
    '''
    x = (std * x) + mean   # unnormalize
    x = x.clamp(0, 1)   # assign less than 0 to 0, bigger than 1 to 1
    x = x.view(x.size(0), c, h, w) # B, C, H, W format for MNIST
    return x


def draw_bbox(img_path, result, color=(255, 0, 0),thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def save_checkpoint(checkpoint_path, model, optimizer, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('models saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, logger, device, optimizer=None):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    logger.info('models loaded from %s' % checkpoint_path)
    return start_epoch


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back
    return newFunc
