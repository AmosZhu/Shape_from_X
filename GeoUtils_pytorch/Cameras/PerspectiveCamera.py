"""
Author: dizhong zhu
Date: 12/08/2022
"""

import torch
import torch.nn as nn
from GeoUtils_pytorch.Geo3D.Rotation import (
    P_f_K_RT
)

from GeoUtils_pytorch.common import (
    make_homegenous
)
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def backProjection(P: torch.Tensor, V: torch.Tensor):
    """
    :param P: camera projection matrices, size=[Batch, 3,4]
    :param V: 3D vertices, size=[Batch,N,3]
    :return:
    """

    # Make vector to homogenous
    V_h = make_homegenous(V)
    v_img = V_h @ P.transpose(1, 2)
    v_img = v_img / v_img[..., 2, None]
    return v_img[..., :2]


class BaseCamera(nn.Module):
    def __init__(self):
        super(BaseCamera, self).__init__()

    def get_camera_center(self):
        pass

    def principal_ray(self):
        """
        Return the principal ray of the camera,
        That's the third row of camera rotation matrix, or third column of camera orientation
        """
        pass

    def point_to_view(self, V: torch.Tensor):
        """
        Convert the input 3D points to camera view coordinates
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 3]
        """
        pass

    def view_to_image(self, V: torch.Tensor):
        """
        Convert the input 3D points to image space. And only apply its camera intrinsic parameters
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 2]
        """
        pass

    def point_to_image(self, V: torch.Tensor):
        """
        Project the 3D points to image space w.r.t camera
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 2]
        """
        pass

    def freeze(self, freeze_list=['']):
        pass

    def upgrade_only(self, unfreeze_list=['']):
        pass


class PerspectiveCamera(BaseCamera):
    def __init__(self,
                 intrinsic: torch.Tensor,
                 rotation: torch.Tensor = None,
                 translation: torch.Tensor = None,
                 scale: torch.Tensor = None
                 ):
        """
        :param Intrinsic: size=[batch,3,3]
        :param Rotation: size=[batch,3,3]
        :param Translation: size=[batch,3]
        :param scale: size=[batch]
        """
        super(PerspectiveCamera, self).__init__()

        nView = intrinsic.shape[0]

        self._K = nn.Parameter(intrinsic)
        self._R = nn.Parameter(torch.eye(3)[None, ...].repeat(nView)) if rotation is None else nn.Parameter(rotation)
        self._t = nn.Parameter(torch.zeros(size=(nView, 3))) if translation is None else nn.Parameter(translation)
        self._s = nn.Parameter(torch.ones(size=(nView), dtype=torch.float32)) if scale is None else nn.Parameter(scale)

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, intrinsic: torch.Tensor):
        self._K = nn.Parameter(intrinsic).to(self._K.device)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, rot_mat: torch.Tensor):
        self._R = nn.Parameter(rot_mat).to(self._R.device)

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, scale: torch.Tensor):
        self._s = nn.Parameter(scale).to(self._s.device)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, trans: torch.Tensor):
        self._t = nn.Parameter(trans).to(self._t.device)

    @property
    def C(self):
        """
        The center of the camera
        """
        return -(self.R.transpose(1, 2) @ self.t[..., None])[..., 0]

    @property
    def P(self):
        """
        The projection matrix of the camera
        :return:
        """
        _P = P_f_K_RT(self.K, self.R, self.t)
        return self.s[..., None, None] * _P

    def freeze(self, freeze_list=['']):
        if 'R' in freeze_list and hasattr(self, '_R'):
            self._R.requires_grad_(False)

        if 'K' in freeze_list and hasattr(self, '_K'):
            self._K.requires_grad_(False)

        if 't' in freeze_list and hasattr(self, '_t'):
            self._t.requires_grad_(False)

        if 's' in freeze_list and hasattr(self, '_s'):
            self._s.requires_grad_(False)

    def upgrade_only(self, unfreeze_list=['']):
        if 'R' not in unfreeze_list and hasattr(self, '_R'):
            self._R.requires_grad_(False)

        if 'K' not in unfreeze_list and hasattr(self, '_K'):
            self._K.requires_grad_(False)

        if 't' not in unfreeze_list and hasattr(self, '_t'):
            self._t.requires_grad_(False)

        if 's' not in unfreeze_list and hasattr(self, '_s'):
            self._s.requires_grad_(False)

    def get_camera_center(self):
        return self.C

    def principal_ray(self):
        """
        Return the principal ray of the camera,
        That's the third row of camera rotation matrix, or third column of camera orientation
        """
        return torch.det(self.R)[..., None] * self.R[:, 2]

    def point_to_view(self, V: torch.Tensor):
        """
        Convert the input 3D points to camera view coordinates
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 3]
        """
        return torch.matmul(self.R, V.transpose(2, 1)).transpose(2, 1) + self.t[:, None, :]

    def view_to_image(self, V: torch.Tensor):
        """
        Convert the input 3D points to image space. And only apply its camera intrinsic parameters
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 2]
        """
        v_img = torch.matmul(self._K, V.transpose(2, 1)).transpose(2, 1)
        v_img = v_img / v_img[..., 2, None]
        return v_img[..., :2]

    def point_to_image(self, V: torch.Tensor):
        """
        Project the 3D points to image space w.r.t camera
        :param V: size=[batch, N, 3]
        :return: size=[batch, N, 2]
        """

        return backProjection(self.P, V)

    def to_dict(self):
        return {'K': self.K.clone(),
                'R': self.R.clone(),
                't': self.t.clone()}

    def __len__(self):
        return self._t.shape[0]

    def __getitem__(self, i):
        return PerspectiveCamera(
            intrinsic=self.K[i][None, ...],
            rotation=self.R[i][None, ...],
            translation=self.t[i][None, ...],
            scale=self.s[i][None, ...]
        )


class ParameterisePerspectiveCamera(PerspectiveCamera):
    def __init__(self,
                 intrinsic: torch.Tensor,
                 rotation: torch.Tensor = None,
                 translation: torch.Tensor = None,
                 scale: torch.Tensor = None
                 ):
        """
        :param Intrinsic: size=[batch,3,3]
        :param Rotation: size=[batch,3,3]
        :param Translation: size=[batch,3]
        """
        super(PerspectiveCamera, self).__init__()

        nView = intrinsic.shape[0]

        _K = intrinsic
        self._fx = nn.Parameter(_K[..., 0, 0])
        self._fy = nn.Parameter(_K[..., 1, 1])
        self._x0 = nn.Parameter(_K[..., 0, 2])
        self._y0 = nn.Parameter(_K[..., 1, 2])
        self._skew = nn.Parameter(_K[..., 0, 1])

        _R = nn.Parameter(torch.eye(3)[None, ...].repeat(nView)) if rotation is None else nn.Parameter(rotation)
        self._R_param = nn.Parameter(matrix_to_quaternion(_R))
        self._t = nn.Parameter(torch.zeros(size=(nView, 3))) if translation is None else nn.Parameter(translation)
        self._s = nn.Parameter(torch.ones(size=(nView,), dtype=torch.float32)) if scale is None else nn.Parameter(scale)

    @property
    def K(self):
        _K = torch.zeros(size=(len(self), 3, 3), dtype=torch.float32).to(self._fx.device)
        _K[..., 0, 0] = self._fx
        _K[..., 1, 1] = self._fy
        _K[..., 0, 2] = self._x0
        _K[..., 1, 2] = self._y0
        _K[..., 0, 1] = self._skew
        _K[..., 2, 2] = 1
        return _K

    @K.setter
    def K(self, intrinsic: torch.Tensor):
        _K = intrinsic
        self._fx = nn.Parameter(_K[..., 0, 0]).to(self._fx.device)
        self._fy = nn.Parameter(_K[..., 1, 1]).to(self._fy.device)
        self._x0 = nn.Parameter(_K[..., 0, 2]).to(self._x0.device)
        self._y0 = nn.Parameter(_K[..., 1, 2]).to(self._y0.device)
        self._skew = nn.Parameter(_K[..., 0, 1]).to(self._skey.device)

    @property
    def R(self):
        return quaternion_to_matrix(self._R_param)

    @R.setter
    def R(self, rot_mat: torch.Tensor):
        self._R_param = nn.Parameter(matrix_to_quaternion(rot_mat)).to(self._R_param.device)

    def freeze(self, freeze_list=['']):
        if 'R' in freeze_list:
            self._R_param.requires_grad_(False)

        if 'f' in freeze_list:
            self._fx.requires_grad_(False)
            self._fy.requires_grad_(False)

        if 'skew' in freeze_list:
            self._skew.requires_grad_(False)

        if 'principal_point' in freeze_list:
            self._x0.requires_grad_(False)
            self._y0.requires_grad_(False)

        remain_list = [e for e in freeze_list if e not in ['R', 'f', 'skew', 'principal_point']]
        super().freeze(remain_list)

    def upgrade_only(self, unfreeze_list=['']):
        if 'R' not in unfreeze_list:
            self._R_param.requires_grad_(False)

        if 'f' not in unfreeze_list:
            self._fx.requires_grad_(False)
            self._fy.requires_grad_(False)

        if 'skew' not in unfreeze_list:
            self._skew.requires_grad_(False)

        if 'principal_point' not in unfreeze_list:
            self._x0.requires_grad_(False)
            self._y0.requires_grad_(False)

        remain_list = [e for e in unfreeze_list if e not in ['R', 'f', 'skew', 'principal_point']]
        super().upgrade_only(remain_list)

    def __getitem__(self, i):
        return ParameterisePerspectiveCamera(
            intrinsic=self.K[i][None, ...],
            rotation=self.R[i][None, ...],
            translation=self.t[i][None, ...],
            scale=self.s[i][None, ...]
        )
