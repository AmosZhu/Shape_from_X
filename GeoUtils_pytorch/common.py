"""
Created by dizhong at 12/11/2020

Some fundamental functions written in pytorch
"""
import torch
import numpy as np


def deg2rad(deg: torch.Tensor):
    return deg * torch.FloatTensor([np.pi]).to(deg.device) / 180


def rad2deg(rad: torch.Tensor):
    return rad * 180 / torch.FloatTensor([np.pi]).to(rad.device)


def euclidian_distance(pt1: torch.Tensor, pt2: torch.Tensor):
    """
    pt1: size=[N,D] N: number of points, D: Dimensions
    pt2: size=[N,D]
    """
    return torch.sqrt(torch.sum((pt1 - pt2) ** 2, dim=-1))


def normalise(input: torch.Tensor, dim=None):
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


# make vectors to homogenous. size=[N,len]
make_homegenous = lambda x: torch.cat((x, torch.ones(size=(*x.shape[:-1], 1), dtype=torch.float32).to(x.device)), dim=-1)

# make a batch vectors to homogenous. size=[batch,N,len]
# make_homegenous3 = lambda x: torch.cat((x, torch.ones(size=(x.shape[0], x.shape[1], 1), dtype=torch.float32).to(x.device)), dim=2)

# make a batch 3x4 matrices to homogenous. size=[batch,3,4]
make_3x4homogenous3 = lambda x: torch.cat([x, torch.FloatTensor([[[0, 0, 0, 1]]]).to(x.device).repeat(x.shape[0], 1, 1)], dim=1)
