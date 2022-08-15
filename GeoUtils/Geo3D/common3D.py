"""
Define some common operations in 3D

numpy version

Created by dizhong at 12/11/2020
"""

import numpy as np


def vec2skew(vec):
    """
    Convert vectors to skew matrix
    :param vec: size=[N,3] or [3]
    :return: size=[N,3,3] or [3,3]
    """
    if vec.ndim == 2:
        assert (vec.shape[1] == 3)

        res = np.zeros(shape=[vec.shape[0], 3, 3], dtype=np.float32)

        res[:, 0, 1] = -vec[..., 2]
        res[:, 0, 2] = vec[..., 1]
        res[:, 1, 0] = vec[..., 2]
        res[:, 1, 2] = -vec[..., 0]
        res[:, 2, 0] = -vec[..., 1]
        res[:, 2, 1] = vec[..., 0]
    elif vec.ndim == 1:
        res = np.array([[0, -vec[2], vec[1]],
                        [vec[2], 0, -vec[0]],
                        [-vec[1], vec[0], 0]])
    else:
        raise ValueError('The input size can only be [3] or [N,3]')

    return res


def skew2vec(m):
    """
    Convert skew matrices to vectors
    :param m: size=[N,3,3] or [3,3]
    :return: size=[N,3] or [3]
    """
    if m.ndim == 3:
        assert (m.shape[1] == 3 and m.shape[2] == 3)
        return np.stack([m[:, 2, 1, None], m[:, 0, 2, None], m[:, 1, 0, None]], dim=1)
    elif m.ndim == 2:
        assert (m.shape[0] == 3 and m.shape[1] == 3)
        return np.array([m[2, 1], m[0, 2], m[1, 0,]])
    else:
        raise ValueError('The input size can only be [N,3,3] or [3,3]')


def cross(v1, v2):
    """
    compute the cross product between two list of vectors
    :param v1: size=[N,3]
    :param v2: size=[N,3]
    :return:
    """

    v3 = np.zeros_like(v1)

    for i in range(v1.shape[0]):
        v3[i] = np.cross(v1[i], v2[i])

    # v3 = np.array([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
    #                v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
    #                v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]]).transpose(1, 0)

    return v3
