"""
Author: dizhong zhu
Date: 12/08/2022
"""

import torch
import torch.nn as nn
from GeoUtils_pytorch.common import (
    make_homegenous
)
from tqdm import tqdm
from .PerspectiveCamera import PerspectiveCamera


class PerspectiveDistortionCamera(PerspectiveCamera):
    def __init__(self, intrinsic: torch.Tensor, rotation: torch.Tensor = None, translation: torch.Tensor = None, radial_distortion: torch.Tensor = None,
                 tangent_distortion: torch.Tensor = None):
        """
        :param Intrinsic: size=[batch,3,3]
        :param Rotation: size=[batch,3,3]
        :param Translation: size=[batch,3]
        :param radia_distortion: size=[batch,3]
        :param tangent_distortion: size=[batch,2]
        :param device: cpu or cuda
        """
        super(PerspectiveDistortionCamera, self).__init__(intrinsic, rotation, translation)

        bs = rotation.shape[0]
        if radial_distortion is not None:
            self._radial = nn.Parameter(radial_distortion.repeat(bs, 1)) if radial_distortion.shape[0] == 1 else nn.Parameter(radial_distortion)
        else:
            self._radial = nn.Parameter(torch.zeros(size=(bs, 3), dtype=torch.float32))

        if tangent_distortion is not None:
            self._tangent = nn.Parameter(tangent_distortion.repeat(bs, 1)) if tangent_distortion.shape[0] == 1 else nn.Parameter(tangent_distortion)
        else:
            self._tangent = nn.Parameter(torch.zeros(size=(bs, 2), dtype=torch.float32))

    @property
    def radial(self):
        return self._radial

    @radial.setter
    def radial(self, raial_distortion):
        self._radial = nn.Parameter(raial_distortion).to(self._radial.device)

    @property
    def tangent(self):
        return self._tangent

    @tangent.setter
    def tangent(self, tangent_distortion):
        self._tangent = nn.Parameter(tangent_distortion).to(self._tangent.device)

    def freeze(self, freeze_list=['']):
        super().freeze(freeze_list)

        if 'radial' in freeze_list:
            self._radial.requires_grad_(False)

        if 'tagent' in freeze_list:
            self._tangent.requires_grad_(False)

    def upgrade_only(self, unfreeze_list=['']):
        super().upgrade_only(unfreeze_list)
        if 'radial' not in unfreeze_list:
            self._radial.requires_grad_(False)

        if 'tagent' not in unfreeze_list:
            self._tangent.requires_grad_(False)

    def point_to_image(self, V: torch.Tensor):
        """
        refer: http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        :param V:
        :return:
        """
        v_view = self.point_to_view(V)
        xy = v_view[..., 0:2] / v_view[..., 2, None]  # Normalise the points

        return self._distortion(xy)

    def view_to_image(self, V: torch.Tensor):
        """
        Convert the input 3D points to image space. And only apply its camera intrinsic parameters
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 2]
        """

        xy = V[..., 0:2] / V[..., 2, None]  # Normalise the points
        return self._distortion(xy)

    def _distortion(self, v: torch.Tensor):
        """
        Distort normalised image point by camera
        :param v: size=[batch, N, 2]
        :return: size=[batch, N ,2]
        """
        k1, k2, k3 = self._radial[..., 0, None], self._radial[..., 1, None], self._radial[..., 2, None]
        p1, p2 = self._tangent[..., 0, None], self._tangent[..., 1, None]

        # distort the pixel in image space
        xy = v
        x, y = xy[..., 0], xy[..., 1]
        r = torch.sqrt(xy[..., 0] ** 2 + xy[..., 1] ** 2)

        xyk = xy * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)[..., None]
        xp = xyk[..., 0] + 2 * p1 * x * y + p2 * (r ** 2 + 2 * x ** 2)
        yp = xyk[..., 1] + p1 * (r ** 2 + 2 * y ** 2) + 2 * p2 * x * y
        xyp = make_homegenous(torch.cat([xp[..., None], yp[..., None]], dim=2))

        # multiply with intrinsic camera matrix
        v_img = torch.matmul(self._K, xyp.transpose(2, 1)).transpose(2, 1)

        return v_img[..., :2]

    def undistortion(self, v: torch.Tensor):
        """
        Undistort the 2D points on the images.
        Because it's a non-linear combination, so use non-linear way to solve it.
        :param v: size=[batch, N, 2]
        :return: size=[batch, N ,2]
        """
        K_inv = torch.inverse(self.K)
        v_h = make_homegenous(v)
        v_n = torch.matmul(K_inv, v_h.transpose(2, 1)).transpose(2, 1)  # Normalised the points

        undistored_point = v_n[..., :2].clone().detach()  # init from the pixel locations
        undistored_point.requires_grad = True

        optimiser = torch.optim.Adam(params=[undistored_point], lr=1e-2)

        max_iteration = 1000
        pre_loss = 0

        n_iter = tqdm(range(max_iteration))

        for _ in n_iter:
            optimiser.zero_grad()

            loss = torch.sum((self._distortion(undistored_point) - v) ** 2, dim=-1)
            loss = torch.sum(torch.mean(loss, dim=-1))

            n_iter.set_description('Unidstorted cameras loss: {0:2f}'.format(loss.data))

            if torch.abs(pre_loss - loss) < 1e-7:
                break

            loss.backward()
            optimiser.step()

        # Transform the normalised point to camera coordinates
        undistored_point_h = make_homegenous(undistored_point.detach())

        v_img = torch.matmul(self.K, undistored_point_h.transpose(2, 1)).transpose(2, 1)

        return v_img[..., :2]

    def to_dict(self):
        return {
            'K': self.K.detach().clone(),
            'R': self.R.detach().clone(),
            't': self.t.detach().clone(),
            'radial_distortion': self.radial.clone(),
            'tangent_distortion': self.tangent.clone()
        }

    def to_dict_list(self):
        return {
            'K': self.K.detach().cpu().numpy().tolist(),
            'R': self.R.detach().cpu().numpy().tolist(),
            't': self.t.detach().cpu().numpy().tolist(),
            'radial_distortion': self.radial.cpu().numpy().tolist(),
            'tangent_distortion': self.tangent.cpu().numpy().tolist()
        }

    def __getitem__(self, i):
        if type(i) in (int,):
            return PerspectiveDistortionCamera(intrinsic=self.K[i][None, ...], rotation=self.R[i][None, ...], translation=self.t[i][None, ...],
                                               radial_distortion=self.radial[i][None, ...], tangent_distortion=self.tangent[i][None, ...])
        else:
            return PerspectiveDistortionCamera(intrinsic=self.K[i], rotation=self.R[i], translation=self.t[i],
                                               radial_distortion=self.radial[i], tangent_distortion=self.tangent[i])
