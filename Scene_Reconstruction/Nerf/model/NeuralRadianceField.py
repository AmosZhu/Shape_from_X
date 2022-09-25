"""
Author: dizhong zhu
Date: 07/04/2022
"""

import torch.nn as nn
from pytorch3d.renderer import (
    RayBundle,
    ray_bundle_to_ray_points,
)
import torch
import math


class fourier_feature_transform(nn.Module):
    def __init__(self, num_basis, max_log_scale=6):
        super(fourier_feature_transform, self).__init__()
        self.num_basis = num_basis
        self.max_log_scale = torch.linspace(0, max_log_scale, num_basis)

    def forward(self, input):
        output = []

        for i, freq in enumerate(self.max_log_scale):
            output.append(torch.cat([torch.sin(2 ** freq * math.pi * input), torch.cos(2 ** freq * math.pi * input)], dim=-1))

        return torch.cat(output, dim=-1)


class BasicNeuralRadianceField(nn.Module):
    '''
    This module without positional encoder
    '''

    def __init__(self, num_features=256, num_layers=1):
        super(BasicNeuralRadianceField, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features=3, out_features=num_features), nn.ReLU(),
                                 *([*nn.Sequential(nn.Linear(in_features=num_features, out_features=num_features), nn.ReLU())] * num_layers))

        self.color_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=3),
            nn.Sigmoid()
        )

        self.density_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1),
            nn.Softplus()
        )

        # self.density_layer[0].bias.data[0] = -1.5

    def forward(self, ray_bundle: RayBundle, **kwargs):
        ray_points_world = ray_bundle_to_ray_points(ray_bundle)
        # ray_points_world = ray_points_world.view(-1, ray_points_world.shape[-1])
        x = self.mlp(ray_points_world)
        color = self.color_layer(x)
        # density = self.density_layer(x)
        density = 1 - (-self.density_layer(x)).exp()
        # rgb = color[..., :3]
        # density = color[..., 3][..., None]
        return density, color

    def batch_forward(self, ray_bundle: RayBundle, n_batches=16, **kwargs):
        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        ray_densities = []
        ray_colors = []
        for batch_idx in batches:
            batch_rays = RayBundle(origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                                   directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                                   lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                                   xys=None)
            density, color = self.forward(batch_rays)
            ray_densities.append(density)
            ray_colors.append(color)

        ray_densities = torch.cat(ray_densities, dim=0).view(*spatial_size, -1)
        ray_colors = torch.cat(ray_colors, dim=0).view(*spatial_size, -1)

        return ray_densities, ray_colors


class FourierNerf(BasicNeuralRadianceField):
    '''
    This model contains positional encoding but without view direction
    '''

    def __init__(self, num_features=256, num_layers=1, num_basis=1, max_log_scale=6):
        super(FourierNerf, self).__init__()

        self.fourier_layer = fourier_feature_transform(num_basis, max_log_scale)

        self.mlp = nn.Sequential(nn.Linear(in_features=num_basis * 2 * 3, out_features=num_features), nn.ReLU(),
                                 *([*nn.Sequential(nn.Linear(in_features=num_features, out_features=num_features), nn.ReLU())] * num_layers))

        self.color_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=3),
            nn.Sigmoid()
        )

        self.density_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1),
            nn.Softplus()
        )

        # self.density_layer[0].bias.data[0] = -1.5

    def forward(self, ray_bundle: RayBundle, **kwargs):
        ray_points_world = ray_bundle_to_ray_points(ray_bundle)

        harmonic_features = self.fourier_layer(ray_points_world)

        # ray_points_world = ray_points_world.view(-1, ray_points_world.shape[-1])
        x = self.mlp(harmonic_features)
        color = self.color_layer(x)
        density = 1 - (-self.density_layer(x)).exp()
        return density, color


class Nerf(nn.Module):
    '''
    Full NERF model
    '''

    def __init__(self, pos_basis=60, view_basis=24, pos_log_scale=10, view_log_scale=4, num_features=256):
        super(Nerf, self).__init__()

        self.mlp4_first = nn.Sequential(nn.Linear(in_features=pos_basis * 2 * 3, out_features=num_features), nn.ReLU(),
                                        *([*nn.Sequential(nn.Linear(in_features=num_features, out_features=num_features), nn.ReLU())] * 3))
        self.mlp4_rest = nn.Sequential(nn.Linear(in_features=pos_basis * 2 * 3 + num_features, out_features=num_features), nn.ReLU(),
                                       *([*nn.Sequential(nn.Linear(in_features=num_features, out_features=num_features), nn.ReLU())] * 3))

        self.density_layer = nn.Sequential(nn.Linear(in_features=num_features, out_features=1), nn.Softplus())
        self.color_layer = nn.Sequential(nn.Linear(in_features=num_features + view_basis * 2 * 3, out_features=128),
                                         nn.ReLU(),
                                         nn.Linear(in_features=128, out_features=3),
                                         nn.Sigmoid())

        self.positional_encoder = fourier_feature_transform(pos_basis, pos_log_scale)
        self.view_encoder = fourier_feature_transform(view_basis, view_log_scale)

        self.density_layer[0].bias.data[0] = -10

    def forward(self, ray_bundle: RayBundle, **kwargs):
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        ray_size = ray_bundle.origins.shape[:-1]

        ray_points_world = ray_bundle_to_ray_points(ray_bundle)
        harmonic_position = self.positional_encoder(ray_points_world)
        harmonic_view = self.view_encoder(ray_bundle.directions)

        # ray_points_world = ray_points_world.view(-1, ray_points_world.shape[-1])
        feature_meadian = self.mlp4_first(harmonic_position)
        feature = self.mlp4_rest(torch.cat([feature_meadian, harmonic_position], dim=-1))

        # density = 1 - (-self.opacity_layer(feature)).exp()

        density = self.density_layer(feature)
        color = self.color_layer(torch.cat([feature, harmonic_view[..., None, :].expand(*ray_size, n_pts_per_ray, harmonic_view.shape[-1])], dim=-1))

        return density, color

    def batch_forward(self, ray_bundle: RayBundle, n_batches=32, **kwargs):
        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        ray_densities = []
        ray_colors = []
        for batch_idx in batches:
            batch_rays = RayBundle(origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                                   directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                                   lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                                   xys=None)
            density, color = self.forward(batch_rays)
            ray_densities.append(density)
            ray_colors.append(color)

        ray_densities = torch.cat(ray_densities, dim=0).view(*spatial_size, -1)
        ray_colors = torch.cat(ray_colors, dim=0).view(*spatial_size, -1)

        return ray_densities, ray_colors
