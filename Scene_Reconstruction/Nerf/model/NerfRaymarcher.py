"""
Author: dizhong zhu
Date: 24/04/2022
"""

import torch
from pytorch3d.renderer.implicit.raymarching import (
    _check_raymarcher_inputs,
    _check_density_bounds,
    _shifted_cumprod
)


class NerfRaymarcher(torch.nn.Module):
    def __init__(self):
        super(NerfRaymarcher, self).__init__()

    def forward(
            self,
            rays_densities: torch.Tensor,
            rays_features: torch.Tensor,
            eps: float = 1e-10,
            **kwargs,
    ):
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.

        Returns:
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        # _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]

        ray_bundle = kwargs.get('ray_bundle')
        depth_samples = ray_bundle.lengths

        # get the depth intervals
        ray_length = ray_bundle.directions.norm(dim=-1, keepdim=True)
        max_depth = kwargs.get('max_depth')

        length_intervals = depth_samples[..., 1:] - depth_samples[..., :-1]  # size=[batch, n_rays, n_sample-1]
        length_intervals = torch.cat([length_intervals, max_depth - depth_samples[..., -1:]], dim=-1)
        dist_intervals = length_intervals * ray_length

        sigma_delta = rays_densities * dist_intervals
        alpha = 1 - (-sigma_delta).exp()

        T = torch.minimum(torch.ones_like(alpha), 1 - alpha + eps)
        T = _shifted_cumprod(T, shift=1)

        weight = (T * alpha)

        # # We predict alpha directly rather than compute
        # alpha = 1 - (-rays_densities).exp()
        # T = _shifted_cumprod((1.0 + eps) - alpha, shift=1)
        #
        # weight = alpha * T

        # opacities = weight.sum(dim=-2)
        features = (rays_features * weight[..., None]).sum(dim=-2)
        opacities = 1.0 - torch.prod(1.0 - alpha, dim=-1, keepdim=True)
        depth = (depth_samples[..., None] * weight[..., None]).sum(dim=-2)

        return torch.cat((features, opacities, depth), dim=-1), weight
