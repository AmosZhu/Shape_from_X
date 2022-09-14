"""
Author: dizhong zhu
Date: 31/05/2021

This will keep updating
operation in 3D that found intersection between different geometries
"""
import torch


def plane_intersect_line(p_normal: torch.tensor, p_point: torch.tensor, p1: torch.tensor, p2: torch.tensor):
    """
    @param p_normal: the normal of the plane.  size=[3]
    @param p_point: a point lie on the plane.  size=[3]
    @param p1, p2: the points define the lines. size=[N,3]

    @return intersect_pt: the intersection point of the line and plane
            intersection_mask: Indicate whether the line and plane intersect
    """

    d = p2 - p1
    d = d / torch.sqrt((d ** 2).sum(-1))[..., None]  # unit vector of directions

    denorm = (d * p_normal[None, ...]).sum(-1)

    s = torch.sum((p_point[None, ...] - p1) * p_normal[None, ...], dim=-1) / denorm

    intersect_pt = p1 + s[..., None] * d

    intersection_mask = ~(denorm == 0)  # if the line perpendicular to the plane normal, it means the plane are parallel to the plane, then never intersect
    # intersect_pt[non_intersect_mask] = None

    return intersect_pt, intersection_mask


def plane_intersect_segment(p_normal: torch.tensor, p_point: torch.tensor, p1: torch.tensor, p2: torch.tensor):
    """
    @param p_normal: the normal of the plane.  size=[3]
    @param p_point: a point lie on the plane.  size=[3]
    @param p1, p2: the end points define the segments. size=[N,3]

    @return intersect_pt: the intersection point of the segement and plane
        intersection_mask: Indicate whether the segement and plane intersect
    """

    p3, p3_mask = plane_intersect_line(p_normal, p_point, p1, p2)

    # we only need to check whether the intersect point p3 is between p1, p2
    v1 = p2 - p1
    v2 = p3 - p1

    dist1 = torch.sqrt(v1 ** 2).sum(dim=-1)
    dist2 = torch.sqrt(v2 ** 2).sum(dim=-1)

    mask1 = torch.sum(v1 * v2, dim=-1) >= 0
    mask2 = dist2 < dist1

    intersection_mask = mask1 & mask2 & p3_mask

    return p3, intersection_mask


def plane_intersect_triangles(p_normal: torch.tensor, p_point: torch.tensor, p1: torch.tensor, p2: torch.tensor, p3: torch.tensor):
    """
    @param p_normal: the normal of the plane.  size=[3]
    @param p_point: a point lie on the plane.  size=[3]
    @param p1, p2, p3: the end points define the triangles. size=[N,3]

    @return intersect_pt: the intersection point of the triangle and plane. size=[N,3,3]
        intersection_mask: Indicate whether the triangle and plane intersect. size=[N,3]
    """

    # test all segments on the triangles

    i1, i1_mask = plane_intersect_segment(p_normal, p_point, p1, p2)
    i2, i2_mask = plane_intersect_segment(p_normal, p_point, p2, p3)
    i3, i3_mask = plane_intersect_segment(p_normal, p_point, p3, p1)

    intersect_pt = torch.cat([i1[:, None, ...], i2[:, None, ...], i3[:, None, ...]], dim=1)
    intersection_mask = torch.cat([i1_mask[..., None], i2_mask[..., None], i3_mask[..., None]], dim=-1)

    return intersect_pt, intersection_mask


def plane_intersect_mesh(p_normal: torch.tensor, p_point: torch.tensor, F: torch.tensor, V: torch.tensor, v_mask: torch.tensor = None):
    """
    @param p_normal: the normal of the plane.  size=[3]
    @param p_point: a point lie on the plane.  size=[3]
    @param F: the face of the mesh             size=[M,3]
    @param V: the vertices of the mesh.        size=[N,3]
    @param mask: the mask of the mesh: 1 indicate the valid vertex on the mesh we want to do intersection. size=[N]

    @ return paths: Array of intersect point
    """

    v_mask = torch.ones(size=[V.shape[0]], dtype=torch.float32).to(F.device) if v_mask is None else v_mask

    # convert the face indices to actuall point in 3D
    FV = V[F]  # size=[M,3,3]
    FV_mask = v_mask[F]  # size=[M,3], only if all vertex mask are one, the face are valid
    F_mask = torch.all(FV_mask, dim=-1)

    FV = FV[F_mask]  # only compute the intersection on valid faces

    # compute the points distance to the plane
    dist = ((FV - p_point[None, None, ...]) * p_normal[None, None, ...]).sum(-1)  # size=[M,3]

    # if the test polygon not intersect with the plane, the sum of sign can only be -3 or 3. Which mean all points locate on one side of the polygon
    intersect_mask = ~(torch.sign(dist).sum(-1).abs() == 3)

    # Take the polygon faces only contains valid intersection between face and planes
    intersect_FV = FV[intersect_mask]

    # compute actuall intersection points between the polygon
    intersect_pt, intersection_mask = plane_intersect_triangles(p_normal, p_point, p1=intersect_FV[:, 0], p2=intersect_FV[:, 1], p3=intersect_FV[:, 2])

    path = intersect_pt[intersection_mask]

    # The mesh triangles will share the edges, so create a lot duplicates, use unique points only
    path = torch.unique(path, dim=0)

    return path
