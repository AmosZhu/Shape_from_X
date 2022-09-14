"""
Author: dizhong zhu
Date: 14/09/2022
"""

import torch
from ..common import (
    make_homegenous
)
from .common3D import (
    vec2skew
)
from .Epipolar import (
    E_f_F,
    Rt_f_E,
)


def triangulateReconstruction(pxs: torch.tensor, cams: torch.tensor, mask: torch.tensor = None):
    '''
    :param pxs: Corresponding pixels in images. size=[nView,N,2]
    :param cams: Camera projection matrices. size=[nView,3,4]
    :param mask: Visible mask to tell whether the pixels contribute to 3D reconstruction. size=[nView,N]
    :return: 3D point cloud cross all view, size=[N,3]
    '''
    noofPoints = pxs.shape[1]
    nView = cams.shape[0]

    mask = torch.ones(size=(noofPoints, nView), dtype=torch.bool).to(pxs.device) if mask is None else mask.transpose(0, 1)

    pxs_h = make_homegenous(pxs)

    pt_x = vec2skew(pxs_h).transpose(0, 1)
    A = pt_x @ cams[None, ...] * mask[..., None, None]
    A = A.view(noofPoints, -1, 4)
    _, _, Vh = torch.linalg.svd(A)
    x = Vh.conj().mT[..., -1]
    pt_3d = x[..., :3] / x[..., 3:]

    return pt_3d


def PoseEstimation_f_F_K(F: torch.tensor, px1: torch.tensor, px2: torch.tensor, K1: torch.tensor, K2: torch.tensor = None):
    '''
    Estimate camera poses, and 3D point cloud from fundamental matrix with known intrinsic parameters
    :param F: Fundamental matrix.                   sisze=[3,3]
    :param px1: image feature in first image.       sisze=[N,2]
    :param px2: image feature in second image.      sisze=[N,2]
    :param K1: intrinsic matrix for first view      sisze=[3,3]
    :param K2: intrinsic matrix for second view, if K2=None, then set K2=K1
    :return: R -> rotation matrix.      size=[3, 3]
             t -> translation vector    size=[3]
             pt-> 3D point cloud of the feature.    size=[N, 3]
    '''

    assert len(F.shape) == 2, 'No batch support!'
    assert px1.shape == px2.shape
    K2 = K1 if K2 is None else K2

    E = E_f_F(F[None], K1[None], K2[None])  # get essential matrix
    Rs, ts = Rt_f_E(E)  # get possible rotation and translation
    Rs = Rs[0]
    ts = ts[0]

    # the correct rotation and translation should give triangulation point set in front of the camera. So we vote the one who give most!
    numNegatives = []
    pt_est = []
    for i in range(Rs.shape[0]):
        Rt1 = torch.cat([torch.eye(3), torch.zeros((3, 1))], dim=-1).to(F.device)
        Rt2 = torch.cat([Rs[i], ts[i][..., None]], dim=-1)
        P1 = K1 @ Rt1
        P2 = K2 @ Rt2
        pt_est1 = triangulateReconstruction(pxs=torch.stack([px1, px2]), cams=torch.stack([P1, P2]))
        # Transfer the point to second view corrdinate

        pt_est2 = (make_homegenous(pt_est1) @ Rt2.mT)

        neg = torch.sum((pt_est1[..., -1] < 0) | (pt_est2[..., -1] < 0))
        numNegatives.append(neg)
        pt_est.append(pt_est1)

    idx = torch.argmin(torch.tensor(numNegatives))

    R = Rs[idx]
    t = ts[idx]
    pt3D = pt_est[idx]

    return R, t, pt3D
