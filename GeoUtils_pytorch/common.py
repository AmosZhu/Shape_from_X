"""
Created by dizhong at 12/11/2020

Some fundamental functions written in pytorch
"""
import torch
import numpy as np


def deg2rad(deg: torch.tensor):
    return deg * torch.FloatTensor([np.pi]).to(deg.device) / 180


def rad2deg(rad: torch.tensor):
    return rad * 180 / torch.FloatTensor([np.pi]).to(rad.device)


def euclidian_distance(pt1: torch.tensor, pt2: torch.tensor):
    """
    pt1: size=[N,D] N: number of points, D: Dimensions
    pt2: size=[N,D]
    """
    return torch.sqrt(torch.sum((pt1 - pt2) ** 2, dim=-1))


def normalise(input: torch.tensor, dim=None):
    """
    Normalise a tensor
    """
    if dim is None:
        denorm = torch.sqrt((input ** 2).sum())
        output = input / denorm
    else:
        denorm = torch.sqrt((input ** 2).sum(dim=dim))
        denorm = denorm.unsqueeze(dim=dim)
        output = input / denorm
    return output


def normalise_transform(input: torch.tensor):
    """
    Reference: Multiple View Geometry in Computer vision(Second Edition). Page 107. section 4.4.4
    Normalised the points to center 0, and average distance to origin is sqrt(dim). i.e. if it's image coordinate, then is sqrt(2), if point cloud then sqrt(3)
    :param input: The points going to be normalised. Size=[batch,N,D]
    :return: normalised point size=[batch,N,D] and transformation matrix size=[batch,D,D]
    """
    assert np.ndim(input) == 3, 'input shape must be [nBatch,N,D]'
    D = input.shape[-1]
    centroid = torch.mean(input, dim=-2)  # centroid of the points
    d_vec = input - centroid  # vector from each points to centroid
    davg = torch.sqrt(torch.sum(d_vec ** 2, dim=-1)).mean(dim=-1)  # average distance to centroid

    s = torch.sqrt(torch.tensor(D)) / davg

    T = torch.eye(D + 1)[None, ...]
    T[:, :D, -1] = -centroid
    T *= s
    T[:, -1, -1] = 1

    norm_pt = make_homegenous(input) @ T.mT

    return T, norm_pt[..., :-1]


# make vectors to homogenous. size=[N,len]
make_homegenous = lambda x: torch.cat((x, torch.ones(size=(*x.shape[:-1], 1), dtype=torch.float32).to(x.device)), dim=-1)

# make a batch 3x4 matrices to homogenous. size=[batch,3,4]
make_3x4homogenous3 = lambda x: torch.cat([x, torch.FloatTensor([[[0, 0, 0, 1]]]).to(x.device).repeat(x.shape[0], 1, 1)], dim=1)


def make_tensor_homogenous(x: torch.tensor):
    tail = torch.zeros(size=(*x.shape[:-2], 1, x.shape[-1])).to(x.device) if x.dim() > 2 else torch.zeros(size=(1, x.shape[-1])).to(x.device)
    tail[..., -1] = 1.0

    x_h = torch.cat([x, tail], dim=-2)

    return x_h
