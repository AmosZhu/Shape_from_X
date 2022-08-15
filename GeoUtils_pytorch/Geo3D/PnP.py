"""

Implementation on Perspective-n-Points algorithm by given different constraints

Author: dizhong zhu
Date: 20/11/2020

"""

import torch
from GeoUtils_pytorch.Geo3D.common3D import (
    skew2vec,
    vec2skew
)

from GeoUtils_pytorch.Geo3D.Rotation import (
    R_cleanUp
)

from GeoUtils_pytorch.common import (
    make_homegenous,
)


def Rt_f_PnP_K(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor):
    """
    By given 2D-3D correspondences, with known intrinisc parameters, reconstrut rotation matrix and translation vector
    :param pt2D: size=[batch,N,2]
    :param pt3D: size=[batch,N,3]
    :param K: size=[batch,3,3] intrinsick parameters
    :return:
    """

    K_inv = torch.pinverse(K)
    pt3Dh = make_homegenous(pt3D)

    bs, N = pt2D.shape[:2]
    A = torch.zeros(size=[bs, N * 2, 12], dtype=torch.float32).to(pt2D.device)

    A[..., :N, 4:8] = -pt3Dh
    A[..., :N, 8:] = pt2D[..., 1][..., None] * pt3Dh
    A[..., N:, :4] = pt3Dh
    A[..., N:, 8:] = -pt2D[..., 0][..., None] * pt3Dh

    _, _, Vh = torch.linalg.svd(A)
    p = Vh.mH[..., -1]

    P = p.view([bs, 3, 4])

    R = torch.matmul(K_inv, P[..., :3])

    U, D, Vh = torch.svd(R)  # make rotation orthogonal
    R_hat = torch.matmul(U, Vh)

    t = (torch.matmul(K_inv, P[..., 3, None]).squeeze() / D[..., 0, None])

    signs = torch.det(R_hat)
    R_hat = signs[..., None, None] * R_hat
    t = signs[..., None] * t

    return R_hat, t


def Rt_f_PnP_K_r(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, r2: torch.tensor, mask=None):
    """
    By given 2D-3D correspondences, with known intrinisc parameters, reconstrut rotation matrix and translation vector.
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2
    :param pt2D: size=[batch,N,2]
    :param pt3D: size=[batch,N,3]
    :param K: size=[batch,3,3] intrinsick parameters
    :param r: size=[batch,3]
    :param mask size=[batch,N] indicates which points are valid
    :return: rotation matrices and translation vectors
    """

    K_inv = torch.pinverse(K)
    pt2Dh = make_homegenous(pt2D)

    V = pt3D
    v = torch.matmul(K_inv, pt2Dh.transpose(2, 1)).transpose(2, 1)
    a = r2[:, None, :]

    bs, N = v.shape[:2]

    mask = torch.ones(size=(bs, N * 2), dtype=torch.float32, device=pt2D.device) if mask is None else mask.repeat(1, 2)

    A = torch.zeros(size=[bs, N * 2, 6], dtype=torch.float32, device=pt2D.device)

    A[:, :N, 0] = V[..., 2] * (a[..., 2] + a[..., 1] * v[..., 1])
    A[:, :N, 1] = -V[..., 0] - V[..., 2] * a[..., 0] * v[..., 1]
    A[:, :N, 2] = V[..., 0] * v[..., 1] - V[..., 2] * a[..., 0]
    A[:, :N, 4] = -torch.ones(N, dtype=torch.float32, device=pt2D.device)
    A[:, :N, 5] = v[..., 1]

    A[:, N:, 0] = V[..., 0] - V[..., 2] * a[..., 1] * v[..., 0]
    A[:, N:, 1] = V[..., 2] * (a[..., 2] + a[..., 0] * v[..., 0])
    A[:, N:, 2] = -V[..., 2] * a[..., 1] - V[..., 0] * v[..., 0]
    A[:, N:, 3] = torch.ones(N, dtype=torch.float32, device=pt2D.device)
    A[:, N:, 5] = -v[..., 0]

    b = torch.cat([V[..., 1] * (a[..., 1] - a[..., 2] * v[..., 1]),
                   -V[..., 1] * (a[..., 0] - a[..., 2] * v[..., 0])], dim=1)

    res = torch.matmul(torch.pinverse(A) * mask[:, None, :], b[..., None] * mask[..., None]).squeeze()
    r1 = res[..., :3]
    r3 = torch.cross(r1, r2)

    R_hat = torch.cat([r1[..., None], r2[..., None], r3[..., None]], dim=2)
    R = R_cleanUp(R_hat)
    t = t_f_PnP_K_R(pt2D, pt3D, K, R)

    return R, t


def Rt_f_PnP_K_r_v2(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, r2: torch.tensor):
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
    K_inv = torch.pinverse(K)
    pt2Dh = make_homegenous(pt2D)

    V = pt3D
    v = torch.matmul(K_inv, pt2Dh.transpose(1, 0)).transpose(1, 0)
    a = r2

    N = v.shape[0]

    A = torch.zeros(size=[N * 2, 6], dtype=torch.float32, device=pt2D.device)

    A[:N, 0] = V[..., 2] * (a[..., 2] + a[..., 1] * v[..., 1])
    A[:N, 1] = -V[..., 0] - V[..., 2] * a[..., 0] * v[..., 1]
    A[:N, 2] = V[..., 0] * v[..., 1] - V[..., 2] * a[..., 0]
    A[:N, 4] = -torch.ones(N, dtype=torch.float32, device=pt2D.device)
    A[:N, 5] = v[..., 1]

    A[N:, 0] = V[..., 0] - V[..., 2] * a[..., 1] * v[..., 0]
    A[N:, 1] = V[..., 2] * (a[..., 2] + a[..., 0] * v[..., 0])
    A[N:, 2] = -V[..., 2] * a[..., 1] - V[..., 0] * v[..., 0]
    A[N:, 3] = torch.ones(N, dtype=torch.float32, device=pt2D.device)
    A[N:, 5] = -v[..., 0]

    b = torch.cat([V[..., 1] * (a[..., 1] - a[..., 2] * v[..., 1]),
                   -V[..., 1] * (a[..., 0] - a[..., 2] * v[..., 0])], dim=0)

    res = torch.lstsq(b[..., None], A)[0].squeeze()
    r1 = res[:3]
    r3 = torch.cross(r1, r2)

    R_hat = torch.cat([r1[..., None], r2[..., None], r3[..., None]], dim=1)
    R = R_cleanUp(R_hat[None, ...])
    t = t_f_PnP_K_R(pt2D[None, ...], pt3D[None, ...], K[None, ...], R)

    return R, t


def Rt_f_multiview_PnP_K_r(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, r2: torch.tensor, miny: torch.tensor, conf: torch.tensor):
    """
    By given 2D-3D correspondences, with known intrinisc parameters, and acceleration vector. Constraint the camera are same height through all veiws
    reconstrut rotation matrix and translation vector.
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2

    :param pt2D: size=[nviews, N, 2,]
    :param pt3D: size=[nviews, N ,3]
    :param K: size=[nviews, 3,3] intrinsic parameters
    :param a: size=[nviews, 3]
    :param miny: size=[nviews,]
    :param conf: size=[nviews,N] the confidence of each pt2D.

    :return: Rs: size=[nviews, 3,3]
    :return: Ts: size=[nviews, 3]
    """
    """ To Do: check the accelreation vector is zero not, might break this code! """

    nviews, N = pt2D.shape[0:2]

    K_inv = torch.pinverse(K)
    pt2Dh = make_homegenous(pt2D)
    vs = torch.matmul(K_inv, pt2Dh.transpose(2, 1)).transpose(2, 1)

    A = torch.zeros(size=[(N * 2 + 1) * nviews, 5 * nviews + 1], dtype=torch.float32, device=pt2D.device)
    b = torch.zeros(size=[(N * 2 + 1) * nviews], dtype=torch.float32, device=pt2D.device)

    # Build linear system for rotations and translations
    for i in range(nviews):
        V = pt3D[i]
        v = vs[i]
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
        A[r + N:r + 2 * N, c + 3] = torch.ones(N, dtype=torch.float32, device=pt2D.device)
        A[r + N:r + 2 * N, c + 4] = -v[..., 0]
        A[r + N:r + 2 * N] *= w[..., None]

        A[r + 2 * N, c:c + 3] = a

        b[r:r + 2 * N + 1] = torch.cat([w * (V[..., 1] * (a[..., 1] - a[..., 2] * v[..., 1]) - miny[i] / a[1]),
                                        w * (-V[..., 1] * (a[..., 0] - a[..., 2] * v[..., 0])),
                                        torch.zeros(size=(1,), device=pt2D.device)])

    # Solve linear system
    res = torch.lstsq(b[..., None], A)[0].squeeze()[:5 * nviews + 1]

    delta_y = res[-1]
    rt = res[:-1].view(-1, 5)
    r1 = rt[:, :3]
    r3 = torch.cross(r1, r2)

    R_hat = torch.cat([r1[..., None], r2[..., None], r3[..., None]], dim=2)
    Rs = R_cleanUp(R_hat)

    tmp = rt[:, 3:]
    t2 = (-delta_y - miny - r2[:, 0] * tmp[:, 0] - r2[:, 2] * tmp[:, 1]) / r2[:, 1]
    Ts = torch.cat([tmp[:, 0, None], t2[..., None], tmp[:, 1, None]], dim=1)

    return Rs, Ts, delta_y


def t_f_PnP_K_R(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, R: torch.tensor):
    """
    By given 2D-3D correspondences, intrinsic matrix and rotation matrix we are aiming to calculate rotation matrices
    :param pt2D: size=[batch,N,2]
    :param pt3D: size=[batch,N,3]
    :param K:    size=[batch,3,3]
    :param R:    size=[batch,3,3]
    :return:     size=[batch,3]
    """
    K_inv = torch.pinverse(K)
    pt2Dh = torch.matmul(K_inv, make_homegenous(pt2D).transpose(2, 1)).transpose(2, 1)

    bs, N = pt2Dh.shape[:2]

    tmp = pt2Dh.contiguous().view(bs * N, -1)
    vx = vec2skew(tmp)
    V = torch.matmul(R, pt3D.transpose(2, 1)).transpose(2, 1).contiguous().view(bs * N, 3, 1)

    b = -torch.matmul(vx, V).view(bs, 3 * N, 1)

    A = vx.view(bs, 3 * N, -1)
    t = torch.matmul(torch.pinverse(A), b).squeeze()

    return t
