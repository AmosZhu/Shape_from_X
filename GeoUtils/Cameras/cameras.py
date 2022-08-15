"""
Author: dizhong zhu
Date: 11/08/2022
"""

from GeoUtils.common import (
    make_homegenous
)


def backProjection(P, V):
    """
    :param P: camera projection matrices, size=[3,4]
    :param V: 3D vertices, size=[N,3]
    :return:
    """

    # Make vector to homogenous
    V_h = make_homegenous(V)
    v_img = V_h @ P.T
    v_img = v_img / v_img[..., 2:]
    return v_img[..., :2]
