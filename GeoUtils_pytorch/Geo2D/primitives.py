"""
Created by dizhong at 12/11/2020

Primitives for projection geometry, pytorch only
2D:
   Lines.
   Points.
   Conics.
"""

import torch


def line2D_f_points(imgpts: torch.Tensor):
    """
    :param imgpts: size=[batch,N,2] where N>=2
    :return: line functions, size=[batch,3]
    """
    make_homogeneous = lambda x: torch.cat((x, torch.ones(size=(x.shape[0], x.shape[1], 1), dtype=torch.float32).to(imgpts.device)), dim=2)
    img_h = make_homogeneous(imgpts)
    _, _, V = torch.svd(img_h)
    l = V[..., -1]
    l = l / torch.norm(l, dim=1)[..., None]
    return l


def vanishpoint_f_lines2D(lines: torch.Tensor):
    """
    :param lines: size=[batch,N,3] where N>=2
    :return: vanish points, size=[batch,2]
    """
    _, _, V = torch.svd(lines)
    v = V[..., -1]
    return v[..., :2] / v[..., 2, None]
