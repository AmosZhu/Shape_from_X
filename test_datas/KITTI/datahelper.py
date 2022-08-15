"""
Author: dizhong zhu
Date: 14/08/2022

load datas from KITTI datas
"""

import cv2
import numpy as np
import os
from PIL import Image

__dirpath__ = os.path.dirname(os.path.abspath(__file__))


def load_data(data_path=None, noofImages=5):
    data_path = __dirpath__ if data_path is None else data_path

    Img_path = f'{data_path}/sequences/00'
    gt_poses_path = f'{data_path}/poses'

    # load camera intrinsic parameters
    with open(f'{Img_path}/calib.txt') as f:
        Ks = []
        for i in range(4):
            K_raw = f.readline().split(':')[1]
            K = np.array(K_raw.split(), dtype=np.float32).reshape(3, 4)[..., :3]
            Ks.append(K)

    Ks = np.stack(Ks)

    # read the ground truth trajectory of the car
    gt_trajectory = {
        'R': [],
        'C': []
    }
    with open(f'{gt_poses_path}/00.txt') as f:
        for i in range(noofImages):
            RC_raw = np.array(f.readline().split(), dtype=np.float32).reshape(3, 4)
            gt_trajectory['R'].append(RC_raw[:, :3])
            gt_trajectory['C'].append(RC_raw[:, 3])

        gt_trajectory['R'] = np.stack(gt_trajectory['R'])
        gt_trajectory['C'] = np.stack(gt_trajectory['C'])

    # load images of left and right images respectively

    img_files = sorted(os.listdir(f'{Img_path}/image_0'))
    imgs_l = []
    imgs_r = []
    for i, img_file in enumerate(img_files):
        if i >= noofImages: break
        img_l = Image.open(f'{Img_path}/image_0/{img_file}')
        img_r = Image.open(f'{Img_path}/image_1/{img_file}')

        imgs_l.append(img_l)
        imgs_r.append(img_r)

    imgs_l = np.stack(imgs_l)
    imgs_r = np.stack(imgs_r)

    return (imgs_l, imgs_r), Ks, gt_trajectory

