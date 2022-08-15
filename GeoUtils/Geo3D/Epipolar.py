"""
Author: dizhong zhu
Date: 10/08/2022

Some function frequently used in epipolar geometry.

If you want to compute fundamental matrix please use opencv package.
"""

import numpy as np
from GeoUtils.Geo3D.common3D import (
    vec2skew,
    skew2vec
)
from GeoUtils.common import (
    normalise
)


def epipoles_f_F(F):
    '''
    return epipole of the fundamental matrix. Which F
    :param F:
    :return: e0,e1 -> Epipole of the fundamental matrix. which Fe0=0. To get the epipole in second view, then it should satisfy F'e1=0
    reference: Multiple view Geometry in Computer vision (Second Edition). Page 246
    '''
    U, D, Vh = np.linalg.svd(F)
    e0 = Vh.conj().T[..., -1]
    e1 = U[..., -1]
    return e0, e1


def P_f_F(F):
    '''
    :param F: Fundamental matrix with size [3x3]
    :return: Projection matrix with size [3x4], P0=[I|0], P1=[ e'xF | e']
    '''
    _, e1 = epipoles_f_F(F)
    e1x = vec2skew(e1)
    P = np.hstack([e1x @ F, e1[..., None]])
    return P


def E_f_F(F, K1, K2=None):
    '''
    Compute essential matrix from fundamental matrix
    :param F: Fundamental matrix
    :param K1: the intrinsic matrix of first camera
    :param K2: the intrinsic matrix of second camera, if None, then set K1=K0
    :return: essential matrix
    '''
    K2 = K1 if K2 is None else K2

    E = K2.T @ F @ K1
    U, _, Vh = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ Vh
    E = normalise(E)

    E = -E if E[2, 2] < 0 else E
    return E


def Rt_f_E(E):
    '''
    Return four possible rotation and translation combination from essential matrix
    :param E:
    :return: R=[4,3,3], t=[4,3]. To get correct R and t, you need to verify the 3D reconstruction points should in front of both cameras
    '''
    U, _, Vh = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    # get possible rotation matrices
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh

    R1 = -R1 if np.linalg.det(R1) < 0 else R1
    R2 = -R2 if np.linalg.det(R2) < 0 else R2

    # get translation
    t = skew2vec(U @ Z @ U.T)

    return np.stack([R1, R1, R2, R2], axis=0), np.stack([t, -t, t, -t], axis=0)
