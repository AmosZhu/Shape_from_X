"""
A numpy version to implement everything about rotation matrices

Created by dizhong at 12/11/2020
"""
import numpy as np
from ..common import (
    deg2rad,
    rad2deg
)


def rotx(deg):
    """
    rotation degree around x axis
    :param deg:
    :return:
    """
    rad = deg2rad(deg)
    sinx = np.sin(rad)
    cosx = np.cos(rad)

    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])

    return Rx


def roty(deg):
    """
    rotation degree around y axis
    :param deg:
    :return:
    """
    rad = deg2rad(deg)
    sinx = np.sin(rad)
    cosx = np.cos(rad)

    Ry = np.array([[cosx, 0, sinx],
                   [0, 1, 0],
                   [-sinx, 0, cosx]])

    return Ry


def rotz(deg):
    """
    rotation degree around z axis
    :param deg:
    :return:
    """
    rad = deg2rad(deg)
    sinx = np.sin(rad)
    cosx = np.cos(rad)

    Rz = np.array([[cosx, -sinx, 0],
                   [sinx, cosx, 0],
                   [0, 0, 1]])

    return Rz


def R_f_twoVec(v1, v2):
    """
    compute the rotation matrix that give R*v1=v2
    :param v1:
    :param v2:
    :return:
    """

    # We can imagine the rotation between two vectors are on a plane, that rotate by angle theta around the axis orthogonal to this plane

    # 1. Compute the rotation only
    s = np.linalg.norm(np.cross(v1, v2))
    c = np.dot(v1, v2)

    G = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # 2. Get the orientation of this plane, create an orthonomal basis
    u = v1
    v = v2 - c * v1
    v /= np.linalg.norm(v)
    w = np.cross(v2, v1)
    w /= np.linalg.norm(w)

    F = np.concatenate([u[..., None], v[..., None], w[..., None]], axis=1)

    # 3. We transfer v1 to the plane coordinate, then rotate, then go back to the original coordinate
    R = F.dot(G).dot(F.T)

    return R


def R_cleanUp(R):
    """
    Give a rotation matrix, return a clean up rotaiton matrix make sure R*RT=I
    :param R:
    :return:
    """
    U, _, Vh = np.linalg.svd(R)
    return U.dot(Vh)


def R_difference(R1, R2):
    """
    Compute the difference angle between two rotation matrices.
    Return: difference angles in degree
    """
    trace = np.trace(R1.T.dot(R2))
    rad = np.arccos((trace - 1) / 2)
    return rad2deg(rad)
