"""
Author: dizhong zhu
Date: 16/12/2021
"""
import numpy as np
from GeoUtils.Geo3D import common3D
from GeoUtils.common import (
    normalise)
from .Primitives import (
    triangle_normal,
    triangle_area
)


def dist_points_to_points(p, q):
    """
    compute the distance between pair points p and q
    :param p: size=[N,3]
    :param q: size=[N,3]
    :return:
    """

    return np.sqrt(np.sum((p - q) ** 2, axis=-1))


def dist_points_to_lines(p, q, c):
    """
    Compute the distance between points (c) and lines (p,q)
    :param p: points on line. Size=[N,3]
    :param q: points on line. Size=[N,3]
    :param c: points to compute the distance between line. Size=[N,3]
    :return: distance between line and point pair, the projection point on the line
    """

    v1 = p - q
    v2 = c - q

    d = normalise(v1, dim=-1)

    # project the v2 on v1 to get the points
    t = np.sum(v2 * d, axis=-1)

    # if 0<=t<=t_max, the the projection point is on the segments, so compute the distance between projection points and c
    projection_point = q + t[..., None] * d
    dist = dist_points_to_points(projection_point, c)

    return dist, projection_point


def dist_points_to_segments(p, q, c):
    """
    Compute the distance between points (c) and segments (p,q)
    :param p: points on line. Size=[N,3]
    :param q: points on line. Size=[N,3]
    :param c: points to compute the distance between line. Size=[N,3]
    :return: distance between line(p,q) and c, and corresponding point
    """

    v1 = p - q
    v2 = c - q

    d = normalise(v1, dim=-1)

    # project the v2 on v1 to get the points
    t = np.sum(v2 * d, axis=-1)
    t_max = np.sqrt(np.sum(v1 ** 2, axis=-1))

    # if 0<=t<=t_max, the the projection point is on the segments, so compute the distance between projection points and c
    projection_point = q + np.expand_dims(t, axis=1) * d
    dist1 = dist_points_to_points(projection_point, c)

    closes_points = projection_point
    # if t<0, then the distance to the segments are the point c and q
    dist2 = dist_points_to_points(q, c)

    # if t>t_max, then the distance to the segments are the point c and p
    dist3 = dist_points_to_points(p, c)

    dist1[t < 0] = dist2[t < 0]
    closes_points[t < 0] = q[t < 0]

    dist1[t > t_max] = dist3[t > t_max]
    closes_points[t > t_max] = p[t > t_max]

    return dist1, closes_points


def dist_points_to_triangles(a, b, c, p):
    """
    triangles(a,b,c). points(p)
    :param a: size=[N,3]
    :param b: size=[N,3]
    :param c: size=[N,3]
    :param p: points to be compute distance from the triangles  size=[N,3]
    :return: a matrix of distance between points and triangles
    """

    normal = triangle_normal(a, b, c)
    area_triangle = triangle_area(a, b, c)

    # connect points c with arbitray points on the triangles, the distance from c to triangle plane will be their product
    ap = p - a
    # bp = p - b
    # cp = p - c

    t = np.sum(ap * normal, axis=-1)

    pp = p + np.expand_dims(t, axis=1) * normal  # projection point on triangle plane
    # compute the distance between target points and projection points
    dist1 = dist_points_to_points(pp, p)

    closes_points = pp
    # check the projection point is within triangle or not
    area_pac = triangle_area(p, a, c)
    area_pab = triangle_area(p, a, b)

    u = area_pab / area_triangle
    v = area_pac / area_triangle
    r = 1 - u - v

    # check the barycentric coordinate satisfy 0<u<1, 0<v<1, 0<r<1. If not, the point is in triangle plane but not within triangle, the distance to the triangle will be one segment
    outside_mask = (u < 0) | (u > 1) | (v < 0) | (v > 1) | (r < 0) | (r > 1)

    dist_seg1, dist_seg1_point = dist_points_to_segments(a, b, p)
    dist_seg2, dist_seg2_point = dist_points_to_segments(b, c, p)
    dist_seg3, dist_seg3_point = dist_points_to_segments(c, a, p)

    dist_segs = np.stack((dist_seg1, dist_seg2, dist_seg3))
    dist_segs_points = np.stack((dist_seg1_point, dist_seg2_point, dist_seg3_point))

    min_idx = np.argmin(dist_segs, axis=0)
    sec_idx = np.linspace(0, len(min_idx) - 1, len(min_idx)).astype(int)
    dist_segs = dist_segs[min_idx, sec_idx]
    dist_seg_point = dist_segs_points[min_idx, sec_idx]

    dist1[outside_mask] = dist_segs[outside_mask]
    closes_points[outside_mask] = dist_seg_point[outside_mask]

    return dist1, closes_points

# def dist_points_to_triangles(a, b, c, p):
#     """
#     triangles(a,b,c). points(p)
#     :param a: size=[N,3]
#     :param b: size=[N,3]
#     :param c: size=[N,3]
#     :param p: points to be compute distance from the triangles  size=[N,3]
#     :return: a matrix of distance between points and triangles
#     """
#
#     N = a.shape[0]
#
#     normal = triangle_normal(a, b, c)
#     area_triangle = triangle_area(a, b, c)
#
#     # connect points p with arbitray points on the triangles, the distance from p to triangle plane will be their product
#     ap = p - a
#     # bp = p - b
#     # cp = p - c
#
#     t = np.sum(ap * normal, axis=-1)
#
#     pp = p + np.expand_dims(t, axis=1) * normal  # projection point on triangle plane
#     # compute the distance between target points and projection points
#     dist1 = dist_points_to_points(pp, p)
#
#     closes_points = pp
#     # check the projection point is within triangle or not
#     area_pac = triangle_area(p, a, c)
#     area_pab = triangle_area(p, a, b)
#
#     u = area_pab / area_triangle
#     v = area_pac / area_triangle
#     r = 1 - u - v
#
#     # check the barycentric coordinate satisfy 0<u<1, 0<v<1, 0<r<1. If not, the point is in triangle plane but not within triangle, the distance to the triangle will be one segment
#     outside_mask = (u < 0) | (u > 1) | (v < 0) | (v > 1) | (r < 0) | (r > 1)
#
#     dist_seg1, dist_seg1_point = dist_points_to_segments(a, b, p)
#     dist_seg2, dist_seg2_point = dist_points_to_segments(b, c, p)
#     dist_seg3, dist_seg3_point = dist_points_to_segments(c, a, p)
#
#     dist_segs = np.stack((dist_seg1, dist_seg2, dist_seg3))
#     dist_segs_points = np.stack((dist_seg1_point, dist_seg2_point, dist_seg3_point))
#
#     # dist_segs_2 = np.zeros(shape=(N,), dtype=np.float32)
#     # dist_seg_points2 = np.zeros(shape=(N, 3), dtype=np.float32)
#     # for i in range(N):
#     #     idx = np.argmin(dist_segs[:, i])
#     #     dist_segs_2[i] = dist_segs[idx, i]
#     #     dist_seg_points2[i] = dist_segs_points[idx, i]
#
#     min_idx = np.argmin(dist_segs, axis=0)
#     sec_idx = np.linspace(0, len(min_idx) - 1, len(min_idx)).astype(int)
#     dist_segs = dist_segs[min_idx, sec_idx]
#     dist_seg_point = dist_segs_points[min_idx, sec_idx]
#
#     dist1[outside_mask] = dist_segs[outside_mask]
#     closes_points[outside_mask] = dist_seg_point[outside_mask]
#
#     return dist1, closes_points
