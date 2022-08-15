"""
Author: dizhong zhu
Date: 10/08/2022

Reconstruct the 3D points
"""

import numpy as np
from GeoUtils.common import (
    make_homegenous
)
from GeoUtils.Geo3D.common3D import (
    vec2skew
)
from GeoUtils.Geo3D.Epipolar import (
    E_f_F,
    Rt_f_E,
)


def triangulateReconstruction(pxs, cams, mask=None):
    '''
    :param pxs: Corresponding pixels in images. size=[nView,N,2]
    :param cams: Camera projection matrices. size=[nView,3,4]
    :param mask: Visible mask to tell whether the pixels contribute to 3D reconstruction. size=[nView,N]
    :return:
    '''
    noofPoints = pxs.shape[1]
    nView = cams.shape[0]
    pt_3d = []

    if mask is None:
        mask = np.ones(shape=(nView, noofPoints), dtype=np.bool)

    pxs_h = make_homegenous(pxs)

    for i in range(noofPoints):
        # A = np.zeros(shape=(nView * 2, 4), dtype=np.float)
        pt_x = vec2skew(pxs_h[:, i])
        A = []
        for j in range(nView):
            if mask[j, i] == True:
                A.append(pt_x[j] @ cams[j])
            # A_sub = pts[j] @ cams[j]
            # A[2 * j:2 * j + 2] = A_sub[:2]
        A = np.concatenate(A)
        _, _, Vh = np.linalg.svd(A)
        x = Vh.conj().T[..., -1]
        pt_3d.append(x[:3] / x[3])

    return np.stack(pt_3d)


def PoseEstimation_f_F_K(F, px1, px2, K1, K2=None):
    '''
    Estimate camera poses, and 3D point cloud from fundamental matrix with known intrinsic parameters
    :param F: Fundamental matrix
    :param px1: image feature in first image
    :param px2: image feature in second image
    :param K1: intrinsic matrix for first view
    :param K2: intrinsic matrix for second view, if K2=None, then set K2=K1
    :return: R -> rotation matrix
             t -> translation vector
             pt-> 3D point cloud of the feature
    '''

    assert px1.shape == px2.shape
    K2 = K1 if K2 is None else K2

    E = E_f_F(F, K1, K2)  # get essential matrix
    Rs, ts = Rt_f_E(E)  # get possible rotation and translation

    numNegatives = []
    pt_est = []
    # the correct rotation and translation should give triangulation point set in front of the camera. So we vote the one who give most!
    for i in range(Rs.shape[0]):
        Rt1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
        Rt2 = np.concatenate([Rs[i], ts[i][..., None]], axis=-1)
        P1 = K1 @ Rt1
        P2 = K2 @ Rt2
        pt_est1 = triangulateReconstruction(pxs=np.stack([px1, px2]), cams=np.stack([P1, P2]))
        # Transfer the point to second view corrdinate

        pt_est2 = make_homegenous(pt_est1) @ Rt2.T

        neg = np.sum((pt_est1[..., -1] < 0) | (pt_est2[..., -1] < 0))
        numNegatives.append(neg)
        pt_est.append(pt_est1)

    idx = np.argmin(numNegatives)

    R = Rs[idx]
    t = ts[idx]
    pt3D = pt_est[idx]

    return R, t, pt3D
