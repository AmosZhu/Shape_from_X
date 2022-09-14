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
    normalise_transform
)


def P_f_PnP(pt2D: torch.tensor, pt3D: torch.tensor):
    """
    By given 2D-3D correspondences, estimate camera projection matrix
    Reference: Multiple View Geometry in Computer vision(Second Edition). Page 179. Equation 7.2
    :param pt2D: size=[nBatch,N,2]
    :param pt3D: size=[nBatch,N,3]
    :return: P: size=[nBatch,3,4]
    """
    assert pt2D.shape[0] == pt3D.shape[0], '2D and 3D should have same length'
    nBatch, N = pt2D.shape[:2]

    T, norm_pt2D = normalise_transform(pt2D)
    U, norm_pt3D = normalise_transform(pt3D)

    A = torch.zeros(size=(nBatch, N * 2, 12), dtype=torch.float32).to(pt2D.device)

    pt3D_h = make_homegenous(norm_pt3D)

    A[:, :N, 4:8] = -pt3D_h
    A[:, :N, 8:] = norm_pt2D[..., 1, None] * pt3D_h
    A[:, N:, :4] = pt3D_h
    A[:, N:, 8:] = -norm_pt2D[..., 0, None] * pt3D_h

    _, _, Vh = torch.linalg.svd(A)
    p = Vh.conj().mT[..., -1]

    P_norm = p.reshape([3, 4])

    P = torch.linalg.pinv(T) @ P_norm @ U
    return P


def Rts_f_P_K(P: torch.tensor, K: torch.tensor):
    """
    By intrinsic camera parameters. estimate camera rotation and translation from projection matrix

    Remember we can P by AP=0 using SVD, so P is up to scale, Because A*(kP)=0 while k is a scale factor.
    Reference: Multiple View Geometry in Computer vision(Second Edition). Page 179. Equation 7.2
    :param P: size=[3,4]
    :param K: size=[3,3]
    :return: R: size=[3,3], t=[3]
    """

    # because P=K[R |t]. So R=K^(-1)P[...,:3]
    K_inv = torch.linalg.pinv(K)
    R = K_inv @ P[..., :3]

    U, D, Vh = torch.linalg.svd(R)
    R_hat = U @ Vh

    # !!!! we choose the determinant for scale here. Because det(R)=D[0]*D[1]*D[2], we can scale by D[0], but instead we use average
    s = torch.pow(torch.prod(D), 1 / 3)
    t = K_inv @ P[..., 3:] / s  # D[0]

    sign = torch.linalg.det(R_hat)
    R_hat *= sign
    t *= sign

    return R_hat, t.squeeze(-1), s


def Rts_f_PnP_K(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor):
    """
    By given 2D-3D correspondences, with known intrinisc parameters, reconstrut rotation matrix and translation vector
    :param pt2D: size=[nViews,N,2]
    :param pt3D: size=[nViews,N,3]
    :param K: size=[nViews,3,3] intrinsick parameters
    :return:
    """

    # K_inv = torch.pinverse(K)
    # pt3Dh = make_homegenous(pt3D)
    #
    # bs, N = pt2D.shape[:2]
    # A = torch.zeros(size=[bs, N * 2, 12], dtype=torch.float32).to(pt2D.device)
    #
    # A[..., :N, 4:8] = -pt3Dh
    # A[..., :N, 8:] = pt2D[..., 1][..., None] * pt3Dh
    # A[..., N:, :4] = pt3Dh
    # A[..., N:, 8:] = -pt2D[..., 0][..., None] * pt3Dh
    #
    # _, _, Vh = torch.linalg.svd(A)
    # p = Vh.mH[..., -1]
    #
    # P = p.view([bs, 3, 4])

    P = P_f_PnP(pt2D, pt3D)

    Rt = torch.pinverse(K) @ P

    U, D, Vh = torch.linalg.svd(Rt[..., :3])  # make rotation orthogonal
    R = U @ Vh

    s = torch.pow(torch.prod(D, dim=-1), 1 / 3)
    R *= torch.linalg.det(R)

    t = t_f_PnP_K_R(pt2D, pt3D, K, R)

    # t = (torch.matmul(K_inv, P[..., 3, None]).squeeze() / D[..., 0, None])
    #
    # signs = torch.det(R_hat)
    # R_hat = signs[..., None, None] * R_hat
    # t = signs[..., None] * t

    return R, t, s


def Rt_f_PnP_K_r(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, r2: torch.tensor, mask=None):
    """
    By given 2D-3D correspondences, with known intrinisc parameters, reconstrut rotation matrix and translation vector.
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2
    :param pt2D: size=[nViews,N,2]
    :param pt3D: size=[nViews,N,3]
    :param K: size=[nViews,3,3] intrinsick parameters
    :param r: size=[nViews,3]
    :param mask size=[nViews,N] indicates which points are valid

    :return: Rs: size=[nViews, 3,3]
    :return: Ts: size=[nViews, 3]

    # ToDo: This one doesn't work because pytorch least square only support two dimension matrix for least square. Need to change in for loop case
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


# def Rt_f_PnP_K_r_v2(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, r2: torch.tensor):
#     """
#     No batch allowed
#     By given 2D-3D correspondences, with known intrinisc parameters, reconstrut rotation matrix and translation vector.
#     We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
#     can be computed by r3=r1xr2
#     :param pt2D: size=[N,2]
#     :param pt3D: size=[N,3]
#     :param K: size=[3,3] intrinsick parameters
#     :param r2: size=[3]
#     :return: rotation matrices and translation vectors
#     """
#     K_inv = torch.pinverse(K)
#     pt2Dh = make_homegenous(pt2D)
#
#     V = pt3D
#     v = torch.matmul(K_inv, pt2Dh.transpose(1, 0)).transpose(1, 0)
#     a = r2
#
#     N = v.shape[0]
#
#     A = torch.zeros(size=[N * 2, 6], dtype=torch.float32, device=pt2D.device)
#
#     A[:N, 0] = V[..., 2] * (a[..., 2] + a[..., 1] * v[..., 1])
#     A[:N, 1] = -V[..., 0] - V[..., 2] * a[..., 0] * v[..., 1]
#     A[:N, 2] = V[..., 0] * v[..., 1] - V[..., 2] * a[..., 0]
#     A[:N, 4] = -torch.ones(N, dtype=torch.float32, device=pt2D.device)
#     A[:N, 5] = v[..., 1]
#
#     A[N:, 0] = V[..., 0] - V[..., 2] * a[..., 1] * v[..., 0]
#     A[N:, 1] = V[..., 2] * (a[..., 2] + a[..., 0] * v[..., 0])
#     A[N:, 2] = -V[..., 2] * a[..., 1] - V[..., 0] * v[..., 0]
#     A[N:, 3] = torch.ones(N, dtype=torch.float32, device=pt2D.device)
#     A[N:, 5] = -v[..., 0]
#
#     b = torch.cat([V[..., 1] * (a[..., 1] - a[..., 2] * v[..., 1]),
#                    -V[..., 1] * (a[..., 0] - a[..., 2] * v[..., 0])], dim=0)
#
#     res = torch.lstsq(b[..., None], A)[0].squeeze()
#     r1 = res[:3]
#     r3 = torch.cross(r1, r2)
#
#     R_hat = torch.cat([r1[..., None], r2[..., None], r3[..., None]], dim=1)
#     R = R_cleanUp(R_hat[None, ...])
#     t = t_f_PnP_K_R(pt2D[None, ...], pt3D[None, ...], K[None, ...], R)
#
#     return R, t


def Rt_f_multiview_PnP_K_r(pt2D: torch.tensor, pt3D: torch.tensor, K: torch.tensor, r2: torch.tensor, miny: torch.tensor, conf: torch.tensor):
    """
    @@This function is specifically design for SMPL model, where a person (pt3D) stand in front of a person, and the person rotate.

    By given 2D-3D correspondences, with known intrinisc parameters, and acceleration vector. Constraint the camera are same height through all veiws
    reconstrut rotation matrix and translation vector.
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2

    :param pt2D: size=[nViews, N, 2]
    :param pt3D: size=[nViews, N ,3]
    :param K: size=[nViews, 3,3] intrinsic parameters
    :param a: size=[nViews, 3]
    :param miny: size=[nViews,]
    :param conf: size=[nViews,N] the confidence of each pt2D.

    :return: Rs: size=[nViews, 3,3]
    :return: Ts: size=[nViews, 3]
    :return deltay: scalar
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
    res = torch.linalg.lstsq(A, b[..., None])[0].squeeze()[:5 * nviews + 1]

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
    :param pt2D: size=[nViews,N,2]
    :param pt3D: size=[nViews,N,3]
    :param K:    size=[nViews,3,3]
    :param R:    size=[nViews,3,3]
    :return:     size=[nViews,3]
    """
    K_inv = torch.pinverse(K)
    pt2Dh = torch.matmul(K_inv, make_homegenous(pt2D).transpose(2, 1)).transpose(2, 1)

    bs, N = pt2Dh.shape[:2]

    tmp = pt2Dh.contiguous().view(bs * N, -1)
    vx = vec2skew(tmp)
    V = torch.matmul(R, pt3D.transpose(2, 1)).transpose(2, 1).contiguous().view(bs * N, 3, 1)

    b = -torch.matmul(vx, V).view(bs, 3 * N, 1)

    A = vx.view(bs, 3 * N, -1)
    t = (torch.pinverse(A) @ b).squeeze(-1)

    return t


def __build_A_f_P(P):
    '''
    Construct a matrix A that help to compute absolute dual qudaric.
    Where:  omega=P*Q*P^T => Ax=b. Where x is the vectorise form of Q. A is the refactor matrix.
    Reference: Multiple view Geometry in Computer vision(Second edtion). 19.3. Page 464. (Table 19.1)
    :param P: projection matrix. size=[nView,3,4]
    :return:  A matrix with size=[nView*4,10]
    '''
    A = torch.zeros(size=(P.shape[0], 4, 10), dtype=torch.float32).to(P.device)
    A[..., 0, 0] = P[..., 0, 0] ** 2 - P[..., 1, 0] ** 2
    A[..., 0, 1] = 2 * (P[..., 0, 0] * P[..., 0, 1] - P[..., 1, 0] * P[..., 1, 1])
    A[..., 0, 2] = 2 * (P[..., 0, 0] * P[..., 0, 2] - P[..., 1, 0] * P[..., 1, 2])
    A[..., 0, 3] = 2 * (P[..., 0, 0] * P[..., 0, 3] - P[..., 1, 0] * P[..., 1, 3])
    A[..., 0, 4] = P[..., 0, 1] ** 2 - P[..., 1, 1] ** 2
    A[..., 0, 5] = 2 * (P[..., 0, 1] * P[..., 0, 2] - P[..., 1, 1] * P[..., 1, 2])
    A[..., 0, 6] = 2 * (P[..., 0, 1] * P[..., 0, 3] - P[..., 1, 1] * P[..., 1, 3])
    A[..., 0, 7] = P[..., 0, 2] ** 2 - P[..., 1, 2] ** 2
    A[..., 0, 8] = 2 * (P[..., 0, 2] * P[..., 0, 3] - P[..., 1, 2] * P[..., 1, 3])
    A[..., 0, 9] = P[..., 0, 3] ** 2 - P[..., 1, 3] ** 2

    A[..., 1, 0] = P[..., 0, 0] * P[..., 1, 0]
    A[..., 1, 1] = P[..., 0, 0] * P[..., 1, 1] + P[..., 0, 1] * P[..., 1, 0]
    A[..., 1, 2] = P[..., 0, 0] * P[..., 1, 2] + P[..., 0, 2] * P[..., 1, 0]
    A[..., 1, 3] = P[..., 0, 0] * P[..., 1, 3] + P[..., 0, 3] * P[..., 1, 0]
    A[..., 1, 4] = P[..., 0, 1] * P[..., 1, 1]
    A[..., 1, 5] = P[..., 0, 1] * P[..., 1, 2] + P[..., 0, 2] * P[..., 1, 1]
    A[..., 1, 6] = P[..., 0, 1] * P[..., 1, 3] + P[..., 0, 3] * P[..., 1, 1]
    A[..., 1, 7] = P[..., 0, 2] * P[..., 1, 2]
    A[..., 1, 8] = P[..., 0, 2] * P[..., 1, 3] + P[..., 0, 3] * P[..., 1, 2]
    A[..., 1, 9] = P[..., 0, 3] * P[..., 1, 3]

    A[..., 2, 0] = P[..., 0, 0] * P[..., 2, 0]
    A[..., 2, 1] = P[..., 0, 0] * P[..., 2, 1] + P[..., 0, 1] * P[..., 2, 0]
    A[..., 2, 2] = P[..., 0, 0] * P[..., 2, 2] + P[..., 0, 2] * P[..., 2, 0]
    A[..., 2, 3] = P[..., 0, 0] * P[..., 2, 3] + P[..., 0, 3] * P[..., 2, 0]
    A[..., 2, 4] = P[..., 0, 1] * P[..., 2, 1]
    A[..., 2, 5] = P[..., 0, 1] * P[..., 2, 2] + P[..., 0, 2] * P[..., 2, 1]
    A[..., 2, 6] = P[..., 0, 1] * P[..., 2, 3] + P[..., 0, 3] * P[..., 2, 1]
    A[..., 2, 7] = P[..., 0, 2] * P[..., 2, 2]
    A[..., 2, 8] = P[..., 0, 2] * P[..., 2, 3] + P[..., 0, 3] * P[..., 2, 2]
    A[..., 2, 9] = P[..., 0, 3] * P[..., 2, 3]

    A[..., 3, 0] = P[..., 1, 0] * P[..., 2, 0]
    A[..., 3, 1] = P[..., 1, 0] * P[..., 2, 1] + P[..., 1, 1] * P[..., 2, 0]
    A[..., 3, 2] = P[..., 1, 0] * P[..., 2, 2] + P[..., 1, 2] * P[..., 2, 0]
    A[..., 3, 3] = P[..., 1, 0] * P[..., 2, 3] + P[..., 1, 3] * P[..., 2, 0]
    A[..., 3, 4] = P[..., 1, 1] * P[..., 2, 1]
    A[..., 3, 5] = P[..., 1, 1] * P[..., 2, 2] + P[..., 1, 2] * P[..., 2, 1]
    A[..., 3, 6] = P[..., 1, 1] * P[..., 2, 3] + P[..., 1, 3] * P[..., 2, 1]
    A[..., 3, 7] = P[..., 1, 2] * P[..., 2, 2]
    A[..., 3, 8] = P[..., 1, 2] * P[..., 2, 3] + P[..., 1, 3] * P[..., 2, 2]
    A[..., 3, 9] = P[..., 1, 3] * P[..., 2, 3]

    return A.view(-1, 10)


def euclidian_rectify_f_P(P):
    '''
    Compute Absolute dual quadirc from Projection matrices.
    Reference: Multiple view Geometry in Computer vision(Second edtion). 19.3. Page 464
    :param P: A set of projection matrices. size=[nBatch, 3 ,4]
    :return: euclidian transformation matrix size=[4,4]
    '''
    assert len(P.shape) == 3, 'Projection matrix size must be [N,3,4]'
    A = __build_A_f_P(P)

    U, D, Vh = torch.linalg.svd(A)
    q = Vh.conj().mT[..., -1]
    q = q / q[-1]
    Q = torch.tensor([[q[0], q[1], q[2], q[3]],
                      [q[1], q[4], q[5], q[6]],
                      [q[2], q[5], q[7], q[8]],
                      [q[3], q[6], q[8], q[9]]]).to(P.device)

    U, D, Vh = torch.linalg.svd(Q)

    # We multiply the scale to the eigen vector to enforce the diagonal as [1,1,1,0]
    s = torch.sqrt(D)
    H = torch.zeros_like(U)

    H[:, 0] = U[:, 0] * s[0]
    H[:, 1] = U[:, 1] * s[1]
    H[:, 2] = U[:, 2] * s[2]
    H[:, 3] = U[:, 3]

    return H


def K_f_PnP_Rt(pt2D: torch.tensor, pt3D: torch.tensor, R: torch.tensor, t: torch.tensor):
    """
    By given 2D-3D correspondences, rotation matrix and translation vectors we are aiming to calculate a single intrinsic matrix.
    Assume all camera has the same intrinsic parameters.
    We formulate the intrinsic matrix as
    K=[f, 0, x_0 || 0, f, y_0 || 0, 0, 1]
    :param pt2D:    size=[nViews,N,2]
    :param pt3D:    size=[nViews,N,3]
    :param R:       size=[nViews,3,3]
    :param t:       size=[nViews,3]
    :return K:      size=[3,3]
    """
    nviews, N = pt2D.shape[0:2]
    device = pt2D.device

    pt3D_view = (pt3D @ R.transpose(2, 1) + t[:, None, :]).view(-1, 3)  # first transform the point cloud to camera coordinate
    pt2D = pt2D.view(-1, 2)

    A = torch.zeros(size=[N * 2 * nviews, 4], dtype=torch.float32, device=device)
    b = torch.zeros(size=[N * 2 * nviews], dtype=torch.float32, device=device)

    NN = N * nviews  # number of points of all views

    A[:NN, 0] = pt3D_view[..., 0]
    A[:NN, 2] = pt3D_view[..., 2]
    A[NN:, 1] = pt3D_view[..., 1]
    A[NN:, 3] = pt3D_view[..., 2]

    b[:NN] = pt3D_view[..., 2] * pt2D[..., 0]
    b[NN:] = pt3D_view[..., 2] * pt2D[..., 1]

    res = torch.linalg.lstsq(A, b[..., None])[0].squeeze()[:5 * nviews + 1]

    fx, fy, x0, y0 = (res)

    K = torch.tensor([[fx, 0, x0],
                      [0, fy, y0],
                      [0, 0, 1]]).to(device)

    return K


def KRt_f_PnP_K_r_AO(pt2D: torch.tensor, pt3D: torch.tensor, K_init: torch.tensor, r2: torch.tensor, miny: torch.tensor, conf: torch.tensor):
    """
    By given 2D-3D correspondences, with initial intrinisc parameters, and acceleration vector. Constraint the camera are same height through all veiws
    Our goal is to reconstrut rotation matrix and translation vector, also improve the intrinsic camera parameters by using alternative optimisation
    We are given the second column of the rotation vector, so we only need to compute the first column of rotation vector r, the third rotation column
    can be computed by r3=r1xr2

    :param pt2D:        size=[nViews, N, 2]
    :param pt3D:        size=[nViews, N ,3]
    :param K_init:      size=[nViews, 3,3]
    :param r2:          size=[nViews, 3] acceleration vectors
    :param miny:        size=[nViews,] minimum position of the height
    :param conf:        size=[nViews,N] the confidence of each pt2D.

    :return: Rs: size=[nViews, 3,3]
    :return: Ts: size=[nViews, 3]
    :return deltay: scalar
    """

    K_p = K_init
    converge = False
    ploss = 1e10
    max_iter = 1000
    # R, t, deltay = Rt_f_multiview_PnP_K_r(pt2D, pt3D, K_init, r2, miny=miny, conf=conf)

    for n in range(max_iter):
        if converge: break
        R, t, deltay = Rt_f_multiview_PnP_K_r(pt2D, pt3D, K_p, r2, miny=miny, conf=conf)

        K_est = K_f_PnP_Rt(pt2D, pt3D, R, t)[None, ...].repeat(R.shape[0], 1, 1)

        loss = (K_est - K_p).abs().sum()

        print(f'[{n}/{max_iter}]K loss: {loss}')

        if abs(loss - ploss) < 1e-1:
            converge = True
        else:
            ploss = loss
            K_p = K_est

    return K_est, R, t, deltay
