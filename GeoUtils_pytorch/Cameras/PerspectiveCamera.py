"""
Author: dizhong zhu
Date: 12/08/2022
"""
from typing import Union
import torch
import torch.nn as nn
from GeoUtils_pytorch.Geo3D.Rotation import (
    P_f_K_RT
)

from GeoUtils_pytorch.common import (
    make_homegenous
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
    def __init__(self,
                 device: Union[str, torch.device] = 'cpu'):
        super(BaseCamera, self).__init__()
        self.device = device

    def to(self, device: Union[str, torch.device] = 'cpu'):
        for k in dir(self):
            if isinstance(getattr(type(self), k, None), property):
                continue  # if it's a property we dont'care

            v = getattr(self, k)
            if k == 'device':
                setattr(self, k, device)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v.to(device))

        return self

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


class PerspectiveCamera(BaseCamera):
    def __init__(self,
                 intrinsic: torch.Tensor = None,
                 rotation: torch.Tensor = None,
                 translation: torch.Tensor = None,
                 scale: torch.Tensor = None,
                 device: Union[str, torch.device] = 'cpu'
                 ):
        """
        :param Intrinsic: size=[batch,3,3]
        :param Rotation: size=[batch,3,3]
        :param Translation: size=[batch,3]
        :param scale: size=[batch]
        """
        super(PerspectiveCamera, self).__init__(device=device)

        if intrinsic is None and \
                rotation is None and \
                translation is None:
            return  # construct an empty camera

        self.__init_cam__(intrinsic=intrinsic,
                          rotation=rotation,
                          translation=translation,
                          scale=scale)

    def __init_cam__(self,
                     intrinsic: torch.Tensor,
                     rotation: torch.Tensor = None,
                     translation: torch.Tensor = None,
                     scale: torch.Tensor = None,
                     ):
        nView = intrinsic.shape[0]
        self.K = intrinsic
        self.R = torch.eye(3)[None, ...].repeat(nView) if rotation is None else rotation
        self.t = torch.zeros(size=(nView, 3)) if translation is None else translation
        self.s = torch.ones(size=(nView,), dtype=torch.float32) if scale is None else scale

        self.to(self.device)

    @property
    def C(self):
        """
        The center of the camera
        """
        return -(self.R.transpose(1, 2) @ self.t[..., None])[..., 0] if len(self) > 0 else None

    @property
    def orientation(self):
        """
        The orientation of the camera is the transpose of the rotaiton
        :return:
        """
        return self.R.transpose(1, 2) if len(self) > 0 else None

    @property
    def P(self):
        """
        The projection matrix of the camera
        :return:
        """
        if len(self) > 0:
            _P = P_f_K_RT(self.K, self.R, self.t)
            return self.s[..., None, None] * _P
        else:
            return None

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

    def add(self,
            rotation: torch.Tensor,
            translation: torch.Tensor,
            scale: torch.Tensor = None,
            intrinsic: torch.Tensor = None,
            ):
        """
        :param Intrinsic: size=[batch,3,3]
        :param Rotation: size=[batch,3,3]
        :param Translation: size=[batch,3]
        :param scale: size=[batch]
        """
        if not hasattr(self, 'K'):
            # construct new camera if it's an empty camera
            self.__init_cam__(intrinsic=intrinsic,
                              rotation=rotation,
                              translation=translation,
                              scale=scale)
            return

        nView = rotation.shape[0]
        self.R = torch.cat([self.R, rotation.to(self.device)])
        self.t = torch.cat([self.t, translation.to(self.device)])

        s = torch.ones(size=(nView), dtype=torch.float32) if scale is None else scale
        self.s = torch.cat([self.s, s.to(self.device)])

        K = self.K[0][None, ...] if intrinsic is None else intrinsic
        self.K = torch.cat([self.K, K.to(self.device)])

    def add_f_numpy(self,
                    rotation,
                    translation,
                    scale=None,
                    intrinsic=None,
                    ):

        nView = rotation.shape[0]
        R = torch.from_numpy(rotation).float()
        t = torch.from_numpy(translation).float()
        s = torch.ones(size=(nView,), dtype=torch.float32) if scale is None else torch.from_numpy(scale).float()
        K = self.K[0][None, ...] if intrinsic is None else torch.from_numpy(intrinsic).float()

        self.add(intrinsic=K,
                 rotation=R,
                 translation=t,
                 scale=s
                 )

    def to_dict(self):
        return {'K': self.K.detach(),
                'R': self.R.detach(),
                't': self.t.detach(),
                's': self.s.detach()
                }

    def to_dict_numpy(self):
        return {'K': self.K.detach().cpu().numpy(),
                'R': self.R.detach().cpu().numpy(),
                't': self.t.detach().cpu().numpy(),
                's': self.s.detach().cpu().numpy()
                }

    def detach(self):
        return PerspectiveCamera(
            intrinsic=self.K.detach(),
            rotation=self.R.detach(),
            translation=self.t.detach(),
            scale=self.s.detach(),
            device=self.device
        )

    def __len__(self):
        return self.t.shape[0] if hasattr(self, 't') else 0

    def __getitem__(self, i):
        return PerspectiveCamera(
            intrinsic=self.K[i][None, ...],
            rotation=self.R[i][None, ...],
            translation=self.t[i][None, ...],
            scale=self.s[i][None, ...]
        )
