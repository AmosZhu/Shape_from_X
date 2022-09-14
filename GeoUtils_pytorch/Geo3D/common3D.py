"""

Author: dizhong zhu
Date: 26/11/2020

"""
import torch


def vec2skew(vec: torch.tensor):
    """
    Convert vectors to skew matrix
    :param vec: size=[*shape,3]
    :return: size=[*shape,3,3]
    """
    assert (vec.shape[-1] == 3)

    res = torch.zeros(size=[*vec.shape[:-1], 3, 3], dtype=torch.float32, device=vec.device)

    res[..., 0, 1] = -vec[..., 2]
    res[..., 0, 2] = vec[..., 1]
    res[..., 1, 0] = vec[..., 2]
    res[..., 1, 2] = -vec[..., 0]
    res[..., 2, 0] = -vec[..., 1]
    res[..., 2, 1] = vec[..., 0]

    return res


def skew2vec(m: torch.tensor):
    """
    Convert skew matrices to vectors
    :param m: size=[*shape,3,3]
    :return: size=[*shape,3]
    """
    assert (m.shape[-1] == 3 and m.shape[-2] == 3)

    return torch.cat([m[..., 2, 1, None], m[..., 0, 2, None], m[..., 1, 0, None]], dim=-1)
