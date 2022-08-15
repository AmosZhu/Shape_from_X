"""

Author: dizhong zhu
Date: 26/11/2020

"""

import numpy as np
from GeoUtils.Geo3D.Rotation import (
    R_cleanUp
)

from GeoUtils.common import (
    make_homegenous,
    normalise_transform
)

from GeoUtils.Geo3D.common3D import (
    vec2skew
)


def P_f_PnP(pt2D, pt3D):
    """
    By given 2D-3D correspondences, estimate camera projection matrix
    Reference: Multiple View Geometry in Computer vision(Second Edition). Page 179. Equation 7.2
    :param pt2D: size=[N,2]
    :param pt3D: size=[N,3]
    :return: P: size=[3,4]
    """
    assert pt2D.shape[0] == pt3D.shape[0], '2D and 3D should have same length'
    N = pt2D.shape[0]

    T, norm_pt2D = normalise_transform(pt2D)
    U, norm_pt3D = normalise_transform(pt3D)

    A = np.zeros(shape=(N * 2, 12), dtype=np.float32)

    pt3D_h = make_homegenous(norm_pt3D)

    A[:N, 4:8] = -pt3D_h
    A[:N, 8:] = norm_pt2D[..., 1, None] * pt3D_h
    A[N:, :4] = pt3D_h
    A[N:, 8:] = -norm_pt2D[..., 0, None] * pt3D_h

    _, _, Vh = np.linalg.svd(A)
    p = Vh.conj().T[..., -1]

    P_norm = p.reshape([3, 4])

    P = np.linalg.pinv(T) @ P_norm @ U
    return P


def Rts_f_P_K(P, K):
    """
    By intrinsic camera parameters. estimate camera rotation and translation from projection matrix

    Remember we can P by AP=0 using SVD, so P is up to scale, Because A*(kP)=0 while k is a scale factor.
    Reference: Multiple View Geometry in Computer vision(Second Edition). Page 179. Equation 7.2
    :param P: size=[3,4]
    :param K: size=[3,3]
    :return: R: size=[3,3], t=[3]
    """

    # because P=K[R |t]. So R=K^(-1)P[...,:3]
    K_inv = np.linalg.pinv(K)
    R = K_inv @ P[..., :3]

    U, D, Vh = np.linalg.svd(R)
    R_hat = U.dot(Vh)

    # !!!! we choose the determinant for scale here. Because det(R)=D[0]*D[1]*D[2], we can scale by D[0], but instead we use average
    s = np.power(np.prod(D), 1 / 3)
    t = K_inv @ P[..., 3] / s  # D[0]

    (R_hat, t) = (-R_hat, -t) if np.linalg.det(R_hat) < 0 else (R_hat, t)

    return R_hat, t, s


def Rts_f_PnP_K(pt2D, pt3D, K):
    """
    By given 2D-3D correspondences and intrinsic matrix,  estimate camera rotation and tranlsation
    :param pt2D: size=[N,2]
    :param pt3D: size=[N,3]
    :param K:    size=[3,3]
    :return: P: size=[3,4]
    """
    P = P_f_PnP(pt2D, pt3D)
    Rt = np.linalg.pinv(K) @ P

    # R = R_cleanUp(Rt[:, :3])

    U, D, Vh = np.linalg.svd(Rt[:, :3])
    R = U.dot(Vh)

    # !!!! we choose the determinant for scale here. Because det(R)=D[0]*D[1]*D[2], we can scale by D[0], but instead we use average
    s = np.power(np.prod(D), 1 / 3)

    # always remember to check the constraints
    R = -R if np.linalg.det(R) < 0 else R

    t = t_f_PnP_K_R(pt2D, pt3D, K, R)

    return R, t, s


def Rt_f_multiview_PnP_K_r(pt2D, pt3D, K, r2, miny, conf):
    """
    By given 2D-3D correspondences, with known intrinisc parameters, and acceleration vector. Constraint the camera are same height through all veiws
    reconstrut rotation matrix and translation vector.
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2

    :param pt2D: size=[nviews, npts, 2]
    :param pt3D: size=[nviews, npts,3,]
    :param K: size=[nviews, 3,3] intrinsic parameters
    :param r2: size=[nviews, 3]
    :param miny: size=[nviews,]
    :param conf: size=[nviews,N]
    :return: Rs: size=[nviews, 3,3]
    :return: Ts: size=[nviews, 3]
    :return: delta_y (scalar) - height of camera above ground plane
    """

    """ To Do: check the accelreation vector is zero not, might break this code! """

    nviews, N = pt2D.shape[0:2]

    Rs = np.zeros(shape=[nviews, 3, 3, ], dtype=np.float32)
    Ts = np.zeros(shape=[nviews, 3], dtype=np.float32)

    A = np.zeros(shape=[(N * 2 + 1) * nviews, 5 * nviews + 1], dtype=np.float32)
    b = np.zeros(shape=[(N * 2 + 1) * nviews], dtype=np.float32)

    # Build linear system for rotations and translations
    for i in range(nviews):
        K_inv = np.linalg.pinv(K[i])
        pt2Dh = make_homegenous(pt2D[i])

        V = pt3D[i]
        v = K_inv.dot(pt2Dh.transpose()).transpose()
        a = r2[i]
        w = conf[i]

        # construct the A by identify the position of (rows,cols)
        r = (N * 2 + 1) * i
        c = i * 5

        A[r:r + N, c] = V[..., 2] * (a[..., 2] + a[..., 1] * v[..., 1])
        A[r:r + N, c + 1] = -V[..., 0] - V[..., 2] * a[..., 0] * v[..., 1]
        A[r:r + N, c + 2] = V[..., 0] * v[..., 1] - V[..., 2] * a[..., 0]
        A[r:r + N, c + 3] = a[0] / a[1]
        A[r:r + N, c + 4] = v[..., 1] + a[2] / a[1]
        A[r:r + N, -1] = 1 / a[1]
        A[r:r + N] *= w[..., None]  # Apply the confidence as weight

        A[r + N:r + 2 * N, c] = V[..., 0] - V[..., 2] * a[..., 1] * v[..., 0]
        A[r + N:r + 2 * N, c + 1] = V[..., 2] * (a[..., 2] + a[..., 0] * v[..., 0])
        A[r + N:r + 2 * N, c + 2] = -V[..., 2] * a[..., 1] - V[..., 0] * v[..., 0]
        A[r + N:r + 2 * N, c + 3] = np.ones(N, dtype=np.float32)
        A[r + N:r + 2 * N, c + 4] = -v[..., 0]
        A[r + N:r + 2 * N] *= w[..., None]

        A[r + 2 * N, c:c + 3] = a

        b[r:r + 2 * N + 1] = np.hstack([w * (V[..., 1] * (a[..., 1] - a[..., 2] * v[..., 1]) - miny[i] / a[1]),
                                        w * (-V[..., 1] * (a[..., 0] - a[..., 2] * v[..., 0])),
                                        0])

    # Solve linear system
    res = np.linalg.lstsq(A, b[..., None], rcond=None)[0].squeeze(-1)
    delta_y = res[-1]

    # get rotation matrices and translation vector
    for i in range(nviews):
        r = 5 * i
        r1 = res[r:r + 3]
        a = r2[i]
        r3 = np.cross(r1, a)
        R_hat = np.hstack([r1[..., None], a[..., None], r3[..., None]])
        Rs[i] = R_cleanUp(R_hat)

        tmp = res[r + 3:r + 5]
        t2 = (-delta_y - miny[i] - a[0] * tmp[0] - a[2] * tmp[1]) / a[1]
        Ts[i] = np.array([tmp[0], t2, tmp[1]])

    return Rs, Ts, delta_y


def Rt_f_PnP_K_r(pt2D, pt3D, K, r2):
    """
    No batch allowed
    By given 2D-3D correspondences, with known intrinisc parameters, reconstrut rotation matrix and translation vector.
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2
    :param pt2D: size=[N,2]
    :param pt3D: size=[N,3]
    :param K: size=[3,3] intrinsick parameters
    :param r2: size=[3]
    :return: rotation matrices and translation vectors
    """
    K_inv = np.linalg.pinv(K)
    pt2Dh = make_homegenous(pt2D)

    V = pt3D
    v = K_inv.dot(pt2Dh.transpose()).transpose()
    a = r2

    N = v.shape[0]

    A = np.zeros(shape=[N * 2 + 1, 6], dtype=np.float32)

    A[:N, 0] = V[..., 2] * (a[..., 2] + a[..., 1] * v[..., 1])
    A[:N, 1] = -V[..., 0] - V[..., 2] * a[..., 0] * v[..., 1]
    A[:N, 2] = V[..., 0] * v[..., 1] - V[..., 2] * a[..., 0]
    A[:N, 4] = -np.ones(N, dtype=np.float32)
    A[:N, 5] = v[..., 1]

    A[N:-1, 0] = V[..., 0] - V[..., 2] * a[..., 1] * v[..., 0]
    A[N:-1, 1] = V[..., 2] * (a[..., 2] + a[..., 0] * v[..., 0])
    A[N:-1, 2] = -V[..., 2] * a[..., 1] - V[..., 0] * v[..., 0]
    A[N:-1, 3] = np.ones(N, dtype=np.float32)
    A[N:-1, 5] = -v[..., 0]

    # Add constraint that r1*a=0
    A[-1, :3] = r2

    b = np.hstack([V[..., 1] * (a[..., 1] - a[..., 2] * v[..., 1]),
                   -V[..., 1] * (a[..., 0] - a[..., 2] * v[..., 0]),
                   0])

    res = np.linalg.lstsq(A, b[..., None], rcond=None)[0].squeeze(-1)
    r1 = res[:3]
    r3 = np.cross(r1, r2)

    R_hat = np.hstack([r1[..., None], r2[..., None], r3[..., None]])
    R = R_cleanUp(R_hat)

    t = t_f_PnP_K_R(pt2D, pt3D, K, R)

    return R, t


def t_f_PnP_K_R(pt2D, pt3D, K, R):
    """
    By given 2D-3D correspondences, intrinsic matrix and rotation matrix we are aiming to calculate rotation matrices
    :param pt2D: size=[N,2]
    :param pt3D: size=[N,3]
    :param K:    size=[3,3]
    :param R:    size=[3,3]
    :return:     size=[3]
    """
    K_inv = np.linalg.pinv(K)
    pt2Dh = K_inv.dot(make_homegenous(pt2D).transpose()).transpose()

    N = pt2Dh.shape[0]

    vx = vec2skew(pt2Dh)
    A = np.zeros(shape=(3 * N, 3 * N), dtype=np.float32)
    for i in range(N):
        A[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = vx[i]

    V = R.dot(pt3D.T).T.reshape(-1)
    b = -A.dot(V)

    t = np.linalg.lstsq(vx.reshape(N * 3, -1), b[..., None], rcond=None)[0].squeeze(-1)

    return t
