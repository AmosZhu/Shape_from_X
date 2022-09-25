"""
Author: dizhong zhu
Date: 23/03/2022
"""
import numpy as np
import torch
import torch.nn as nn
import math


class fourier_feature_transform(nn.Module):
    def __init__(self, num_basis, max_log_scale=6, fine_iterations=None):
        super(fourier_feature_transform, self).__init__()
        self.num_basis = num_basis
        self.max_log_scale = torch.linspace(0, max_log_scale, num_basis)
        self.fine_iterations = fine_iterations

        if fine_iterations is not None:
            self.K = torch.arange(num_basis, dtype=torch.float32)

    def forward(self, input, n_iter=None):
        output = []

        if self.fine_iterations is not None and n_iter is not None:
            alpha = n_iter / self.fine_iterations * self.num_basis
            running_weight = (1 - ((alpha - self.K).clamp_(min=0, max=1) * np.pi).cos()) / 2
        else:
            running_weight = torch.ones(self.num_basis)

        for i, freq in enumerate(self.max_log_scale):
            w = running_weight[i]
            output.append(w * torch.cat([torch.sin(2 ** freq * math.pi * input), torch.cos(2 ** freq * math.pi * input)], dim=-1))

        return torch.cat(output, dim=-1)


