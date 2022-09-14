"""
Author: dizhong zhu
Date: 14/09/2022

Some function frequently used in epipolar geometry.

If you want to compute fundamental matrix please use opencv package.
"""

import torch
from GeoUtils_pytorch.Geo3D.common3D import (
    vec2skew,
    skew2vec
)
from GeoUtils_pytorch.common import (
    normalise
)


def epipoles_f_F(F: torch.tensor):
    '''
    return epipole of the fundamental matrix. Which F
    :param F: size=[nBatch,3,3]
    :return: e0,e1 -> Epipole of the fundamental matrix. which Fe0=0. To get the epipole in second view, then it should satisfy F'e1=0
    reference: Multiple view Geometry in Computer vision (Second Edition). Page 246
    '''
    U, D, Vh = torch.linalg.svd(F)
    e0 = Vh.conj().mT[..., -1]
    e1 = U[..., -1]
    return e0, e1


def P_f_F(F: torch.tensor):
    '''
    :param F: Fundamental matrix with size=[nBatch,3,3]
    :return: Projection matrix with size=[nBatch,3,4], P0=[I|0], P1=[ e'xF | e']
    '''
    _, e1 = epipoles_f_F(F)
    e1x = vec2skew(e1)
    P = torch.cat([e1x @ F, e1[..., None]], dim=-1)
    return P


def E_f_F(F: torch.tensor, K1: torch.tensor, K2: torch.tensor = None):
    '''
    Compute essential matrix from fundamental matrix
    :param F: Fundamental matrix, size=[nBatch,3,3]
    :param K1: the intrinsic matrix of first camera, size=[nBatch,3,3]
    :param K2: the intrinsic matrix of second camera, size=[nBatch,3,3]. if None, then set K1=K0
    :return: essential matrix
    '''
    K2 = K1 if K2 is None else K2

    E = K2.mT @ F @ K1
    U, _, Vh = torch.linalg.svd(E)
    E = U @ torch.diag(torch.tensor([1, 1, 0], dtype=torch.float32).to(F.device)) @ Vh
    E = normalise(E)

    E = -E if E[:, 2, 2] < 0 else E
    return E


def Rt_f_E(E: torch.tensor):
    '''
    Return four possible rotation and translation combination from essential matrix
    :param E:
    :return: R=[nBatch,4,3,3], t=[nBatch,4,3]. To get correct R and t, you need to verify the 3D reconstruction points should in front of both cameras
    '''
    U, _, Vh = torch.linalg.svd(E)
    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32).to(E.device)[None]
    Z = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=torch.float32).to(E.device)[None]

    # get possible rotation matrices
    R1 = U @ W @ Vh
    R2 = U @ W.mT @ Vh

    R1 = R1 * torch.sign(torch.linalg.det(R1))
    R2 = R2 * torch.sign(torch.linalg.det(R2))

    # get translation
    t = skew2vec(U @ Z @ U.mT)

    return torch.stack([R1, R1, R2, R2], dim=1), torch.stack([t, -t, t, -t], dim=1)
