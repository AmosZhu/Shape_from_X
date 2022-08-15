"""
Author: dizhong zhu
Date: 14/08/2022
"""

from PIL import Image
import cv2
import numpy as np
import os

__dirpath__ = os.path.dirname(os.path.abspath(__file__))


def load_data(noofImage=5):
    with open(f'{__dirpath__}/templeR_par.txt') as f:
        N = int(f.readline())

        noofImage = N if noofImage > N else noofImage

        print(f'There are {noofImage} images in total')
        Rs = []
        ts = []
        Ks = []
        images = []
        for i in range(noofImage):
            cam_raw = f.readline().split()
            file_name = cam_raw[0]
            # img = cv2.imread(f'{__dirpath__}/{file_name}')
            img = Image.open(f'{__dirpath__}/{file_name}').convert('RGB')
            images.append(img)
            # images.append(Image.open(f'{data_path}/{file_name}').convert('RGB'))
            K = np.array(cam_raw[1:10], dtype=np.float32).reshape(3, 3)
            R = np.array(cam_raw[10:19], dtype=np.float32).reshape(3, 3)
            t = np.array(cam_raw[19:], dtype=np.float32)
            Ks.append(K)
            Rs.append(R)
            ts.append(t)

        images = np.stack(images)
        Ks = np.stack(Ks)
        Rs = np.stack(Rs)
        ts = np.stack(ts)
        return images, (Ks, Rs, ts)
