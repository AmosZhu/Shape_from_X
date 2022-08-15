"""
All rotation implementation in pytorch manner

Author: dizhong zhu
Date: 12/11/2020

"""

import torch
from ..common import (
    deg2rad,
    rad2deg
)


def rotx(deg: torch.Tensor):
    """
    :param deg: rotate on y-axis by deg. size=[Batch]
    :return: [Batch, 3, 3] rotation matrix
    """
    Rx = torch.zeros(size=[deg.shape[0], 3, 3], dtype=torch.float32).to(deg.device)

    rad = deg2rad(deg)
    sinx = torch.sin(rad)
    cosx = torch.cos(rad)

    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cosx
    Rx[:, 1, 2] = -sinx
    Rx[:, 2, 1] = sinx
    Rx[:, 2, 2] = cosx

    return Rx


def roty(deg: torch.Tensor):
    """
    :param deg: rotate on y-axis by deg. size=[Batch]
    :return: [Batch, 3, 3] rotation matrix
    """
    Ry = torch.zeros(size=[deg.shape[0], 3, 3], dtype=torch.float32).to(deg.device)

    rad = deg2rad(deg)
    sinx = torch.sin(rad)
    cosx = torch.cos(rad)

    Ry[:, 0, 0] = cosx
    Ry[:, 0, 2] = sinx
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sinx
    Ry[:, 2, 2] = cosx

    return Ry


def rotz(deg: torch.Tensor):
    """
    :param deg: rotate on y-axis by deg. size=[Batch]
    :return: [Batch, 3, 3] rotation matrix
    """
    Rz = torch.zeros(size=[deg.shape[0], 3, 3], dtype=torch.float32).to(deg.device)

    rad = deg2rad(deg)
    sinx = torch.sin(rad)
    cosx = torch.cos(rad)

    Rz[:, 0, 0] = cosx
    Rz[:, 0, 1] = -sinx
    Rz[:, 1, 0] = sinx
    Rz[:, 1, 1] = cosx
    Rz[:, 2, 2] = 1

    return Rz


def R_cleanUp(R: torch.Tensor):
    """
    Give a rotation matrix, return a clean up rotaiton matrix make sure R*RT=I
    :param R:
    :return:
    """
    U, _, Vh = torch.linalg.svd(R)
    return U @ Vh


def R_f_vanishPoints(vp: torch.Tensor, K: torch.Tensor, signs=None):
    """
    given two orthogonal vanish points on the ground floor, the first one
    look at the z direction while the second one look at x direction.
    with given intrinsic camera paramters to compute the camera rotations
    :param vp: vanish points, size=[batch,2,2]
    :param K: intrinsic matrices, size=[batch, 3,3]
    :return: rotation matrices, size=[batch,3,3]
    """
    make_homegenous = lambda x: torch.cat((x, torch.ones(size=(x.shape[0], 1), dtype=torch.float32).to(vp.device)), dim=1)
    signs = torch.ones(size=(vp.shape[0], 2)).to(vp.device) if signs is None else signs

    K_inv = torch.pinverse(K)
    v1 = make_homegenous(vp[:, 0, :])  # vanish points look at z directions
    r3 = signs[..., 0, None, None] * torch.matmul(K_inv, v1[..., None])

    v2 = make_homegenous(vp[:, 1, :])  # vanish points look at x directions
    r1 = signs[..., 1, None, None] * torch.matmul(K_inv, v2[..., None])

    r2 = torch.cross(r3, r1)
    R = torch.cat([r1, r2, r3], dim=2)

    # Clean up rotation

    return R_cleanUp(R)


def P_f_K_RT(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    """
    :param K: Intrisic matrix, size=[Batch, 3,3]
    :param R: Rotation matrix, size=[Batch, 3,3]
    :param t: Translation vector, size=[Batch,3]
    :return: Projection matrix. size=[Batch, 3, 4]
    """
    Rt = torch.cat((R, t[..., None]), dim=-1)
    P = K @ Rt
    return P


def R_difference(R1, R2):
    """
    Compute the difference angle between two rotation matrices.
    Return: difference angles in degree
    """
    trace = torch.trace(R1.T.dot(R2))
    rad = torch.arccos((trace - 1) / 2)

    return rad2deg(rad)
