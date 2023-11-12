"""
Author: dizhong zhu
Date: 31/03/2022
"""
import os

import cv2
import numpy as np
import torch

from torch.utils.data import (
    Dataset
)

__dirpath__ = os.path.dirname(os.path.abspath(__file__))


class matajohDataset(Dataset):
    """
    Load nerf data with one object
    """

    def __init__(self, name, mode='train'):
        self.load(name, mode)

    def load(self, name, mode):
        filename = os.path.join(__dirpath__, 'data', name)
        # if os.path.exists(filename):
        #     # download the data
        #     pass

        data = np.load(filename)
        test_end, height, width = data["images"].shape[:3]
        split_counts = data["split_counts"]
        train_end = split_counts[0]
        val_end = train_end + split_counts[1]

        if mode == "train":
            idx = list(range(train_end))
        elif mode == "val":
            idx = list(range(train_end, val_end))
        elif mode == "test":
            idx = list(range(val_end, test_end))
        else:
            print("Unrecognized mode:", mode)
            return None

        # self.max_depth = np.max(data['bounds']) * 2
        self.images = data["images"][idx].astype(np.float32) / 255.0
        self.intrinsics = data["intrinsics"][idx]
        self.extrinsics = data["extrinsics"][idx]

        self.height, self.width = self.images.shape[1:3]

    def __getitem__(self, item):
        K = self.intrinsics[item]

        Rt = np.linalg.inv(self.extrinsics[item])

        R = Rt[:3, :3]
        t = Rt[:3, 3]

        images = self.images[item]

        return (K, R, t), (images[..., :3].transpose(2, 0, 1), images[..., 3][None, ...])

    def __len__(self):
        return self.images.shape[0]

    def random_select(self, num=1):
        indices=torch.randperm(num)[:num]
        for idx in indices:
            (K, R, t), (image, mask) = self.__getitem__(idx)


    def get_all_data(self):
        K = torch.from_numpy(self.intrinsics)

        Rt = torch.inverse(torch.from_numpy(self.extrinsics))
        R = Rt[..., :3, :3]
        t = Rt[..., :3, 3]

        images = torch.from_numpy(self.images)

        return (K, R, t), (images[..., :3], images[..., 3])

    def get_camera_matrices(self):
        return self.extrinsics

    def max_depth(self):
        cam_center = self.get_camera_matrices()[:, :3, 3]
        return (cam_center.max() - cam_center.min()) / 2

    def save_all_images(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for i, img in enumerate(self.images):
            img = cv2.cv2.cvtColor(img, cv2.cv2.COLOR_RGB2BGR)
            img = img * 255
            img = img.astype(np.uint8)
            cv2.imwrite(f'{path}/{i}.png', img)
