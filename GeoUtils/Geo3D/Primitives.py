"""
Author: dizhong zhu
Date: 16/12/2021
"""
import numpy as np
from GeoUtils.common import (
    normalise
)
from GeoUtils.Geo3D import common3D


def triangle_area(a, b, c):
    """
    a,b,c define the vertices of the triangles
    :param a: size=[N,3]
    :param b: size=[N,3]
    :param c: size=[N,3]
    :return: the area of the triangles
    """

    v1 = b - a
    v2 = c - b

    # v3 is the cross product between v1 and v2.
    v3 = common3D.cross(v1, v2)
    area = np.sqrt(np.sum(v3 ** 2, axis=-1))

    return area / 2


def triangle_normal(a, b, c):
    """
    a,b,c define the vertices of the triangles
    :param a: size=[N,3]
    :param b: size=[N,3]
    :param c: size=[N,3]
    :return: the normal of the triangle, the computation order is a->b->c
    """
    v1 = normalise(b - a, dim=1)
    v2 = normalise(c - b, dim=1)

    v3 = common3D.cross(v1, v2)
    return normalise(v3, dim=1)


def triangle_centroid(a, b, c):
    """
    a,b,c define the vertices of the triangles
    :param a: size=[N,3]
    :param b: size=[N,3]
    :param c: size=[N,3]
    :return: the centroid of each triangles
    """
    return (a + b + c) / 3
