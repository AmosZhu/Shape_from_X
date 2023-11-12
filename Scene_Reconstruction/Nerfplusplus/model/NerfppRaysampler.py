"""
Author: dizhong zhu
Date: 05/10/2022
"""

import torch
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import RayBundle
from torch.nn import functional as F
from GeoUtils_pytorch.Geo3D.Intersection import (
    ray_intersect_sphere
)


def max_depth_ray_intersect_unit_sphere(ray_o, ray_d):
    sphere_o = torch.zeros_like(ray_o).to(ray_o.device)
    sphere_r = torch.ones(ray_o.shape[:-1]).to(ray_o.device)

    t, mask = ray_intersect_sphere(ray_o, ray_d, sphere_o, sphere_r)

    max_depth = torch.max(t, dim=-1)[0]

    return max_depth


class SphereMonteCarloRaysampler(torch.nn.Module):
    """
    This is a tweak from pytorch3D library, that fit the ray sampler is within the sphere for nerf++

    Samples a fixed number of pixels within denoted xy bounds uniformly at random.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.

    For practical purposes, this is similar to MultinomialRaysampler without a mask,
    however sampling at real-valued locations bypassing replacement checks may be faster.
    """

    def __init__(
            self,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            n_rays_per_image: int,
            n_pts_per_ray: int,
            min_depth: float,
            *,
            unit_directions: bool = False,
            stratified_sampling: bool = False,
    ) -> None:
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel.
            max_x: The largest x-coordinate of each ray's source pixel.
            min_y: The smallest y-coordinate of each ray's source pixel.
            max_y: The largest y-coordinate of each ray's source pixel.
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            unit_directions: whether to normalize direction vectors in ray bundle.
            stratified_sampling: if set, performs stratified sampling in n_pts_per_ray
                bins for each ray; otherwise takes n_pts_per_ray deterministic points
                on each ray with uniform offsets.
        """
        super().__init__()
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._unit_directions = unit_directions
        self._stratified_sampling = stratified_sampling

    def forward(
            self, cameras: CamerasBase, *, stratified_sampling: bool = False, **kwargs
    ) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            stratified_sampling: if set, performs stratified sampling in n_pts_per_ray
                bins for each ray; otherwise takes n_pts_per_ray deterministic points
                on each ray with uniform offsets.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        rays_xy = torch.cat(
            [
                torch.rand(
                    size=(batch_size, self._n_rays_per_image, 1),
                    dtype=torch.float32,
                    device=device,
                )
                * (high - low)
                + low
                for low, high in (
                (self._min_x, self._max_x),
                (self._min_y, self._max_y),
            )
            ],
            dim=2,
        )

        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        return _xy_to_ray_bundle_insphere(
            cameras,
            rays_xy,
            self._min_depth,
            self._n_pts_per_ray,
            self._unit_directions,
            stratified_sampling,
        )


def _jiggle_within_stratas(bin_centers: torch.Tensor) -> torch.Tensor:
    """
    Performs sampling of 1 point per bin given the bin centers.

    More specifically, it replaces each point's value `z`
    with a sample from a uniform random distribution on
    `[z - delta_−, z + delta_+]`, where `delta_−` is half of the difference
    between `z` and the previous point, and `delta_+` is half of the difference
    between the next point and `z`. For the first and last items, the
    corresponding boundary deltas are assumed zero.

    Args:
        `bin_centers`: The input points of size (..., N); the result is broadcast
            along all but the last dimension (the rows). Each row should be
            sorted in ascending order.

    Returns:
        a tensor of size (..., N) with the locations jiggled within stratas/bins.
    """
    # Get intervals between bin centers.
    mids = 0.5 * (bin_centers[..., 1:] + bin_centers[..., :-1])
    upper = torch.cat((mids, bin_centers[..., -1:]), dim=-1)
    lower = torch.cat((bin_centers[..., :1], mids), dim=-1)
    # Samples in those intervals.
    jiggled = lower + (upper - lower) * torch.rand_like(lower)
    return jiggled


def _xy_to_ray_bundle_insphere(
        cameras: CamerasBase,
        xy_grid: torch.Tensor,
        min_depth: float,
        n_pts_per_ray: int,
        unit_directions: bool,
        stratified_sampling: bool = False,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.

    Args:
        cameras: cameras object representing a batch of cameras.
        xy_grid: torch.tensor grid of image xy coords.
        min_depth: The minimum depth of each ray-point.
        max_depth: The maximum depth of each ray-point.
        n_pts_per_ray: The number of points sampled along each ray.
        unit_directions: whether to normalize direction vectors in ray bundle.
        stratified_sampling: if set, performs stratified sampling in n_pts_per_ray
            bins for each ray; otherwise takes n_pts_per_ray deterministic points
            on each ray with uniform offsets.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()  # pyre-ignore

    # make two sets of points at a constant depth=1 and 2
    to_unproject = torch.cat(
        (
            xy_grid.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject, from_ndc=True)

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world
    if unit_directions:
        rays_directions_world = F.normalize(rays_directions_world, dim=-1)

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    #####################################################################################
    # The max depth is the intersection point with the unit sphere.
    # Notice it only works the camera has benn normalised, and all cameras settle inside the sphere.
    #####################################################################################
    # Compute the intersection point with sphere

    max_depth = max_depth_ray_intersect_unit_sphere(rays_origins_world, rays_directions_world)

    rays_zs = xy_grid.new_empty((0,))
    if n_pts_per_ray > 0:
        depths = torch.linspace(
            min_depth,
            4.0,
            n_pts_per_ray,
            dtype=xy_grid.dtype,
            device=xy_grid.device,
        )
        rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray)

        if stratified_sampling:
            rays_zs = _jiggle_within_stratas(rays_zs)

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )
