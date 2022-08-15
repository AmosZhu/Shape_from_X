"""
Basic functions for

Author: dizhong zhu
Date: 12/11/2020

"""

import numpy as np


def deg2rad(deg):
    return deg * np.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / np.pi


def euclidian_distance(pt1, pt2):
    """
    pt1: size=[N,D] N: number of points, D: Dimensions
    pt2: size=[N,D]
    """
    return np.sqrt(np.sum((pt1 - pt2) ** 2, axis=1))


def normalise(input, dim=None):
    """
    :param input: the vector to be normalised
    :return:
    """
    if dim is None:
        denorm = np.sqrt((input ** 2).sum())
        output = input / denorm
    else:
        denorm = np.sqrt(np.sum(input ** 2, axis=dim))
        denorm = np.expand_dims(denorm, axis=dim)
        output = input / denorm
    return output


def normalise_transform(input):
    """
    Reference: Multiple View Geometry in Computer vision(Second Edition). Page 107. section 4.4.4
    Normalised the points to center 0, and average distance to origin is sqrt(dim). i.e. if it's image coordinate, then is sqrt(2), if point cloud then sqrt(3)
    :param input: The points going to be normalised. Size=[N,D]
    :return: normalised point and transformation matrix
    """
    assert np.ndim(input) == 2, 'input shape must be [N,D]'
    _, D = input.shape
    centroid = np.mean(input, axis=0)  # centroid of the points
    d_vec = input - centroid  # vector from each points to centroid
    davg = np.sqrt(np.sum(d_vec ** 2, -1)).mean()  # average distance to centroid

    s = np.sqrt(D) / davg

    T = np.eye(D + 1)
    T[:D, -1] = -centroid
    T *= s
    T[-1, -1] = 1

    norm_pt = make_homegenous(input) @ T.T

    return T, norm_pt[..., :-1]


# make vectors to homogenous. size=[N,len]
make_homegenous = lambda x: np.concatenate((x, np.ones(shape=(*x.shape[:-1], 1), dtype=np.float32)), axis=-1)

# # make a batch vectors to homogenous. size=[batch,N,len]
# make_homegenous3 = lambda x: np.hstack((x, np.ones(shape=(x.shape[0], x.shape[1], 1), dtype=np.float32)))
