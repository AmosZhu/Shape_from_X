"""
Author: dizhong zhu
Date: 16/12/2021

1. convert mesh to SDF storing in voxels
"""

from types import SimpleNamespace
import numpy as np
from .Distances import dist_points_to_triangles
from .Primitives import (triangle_normal)


class voxels:
    '''
    Define a square voxel presentation
    '''

    def __init__(self, x_dim=256, y_dim=None, z_dim=None,
                 x_lim=[-1, 1], y_lim=None, z_lim=None):
        """
        :param x_dim: Number of voxel in x dimension
        :param y_dim: Number of voxel in y dimension
        :param z_dim: Number of voxel in z dimension
        :param x_lim: The metric boundary in x dimension, i.e. x_lim=[-1,1] means the voxels in x dimension start from -1m and end in 1m in metric physic world
        :param y_lim: The metric boundary in y dimension
        :param z_lim: The metric boundary in z dimension
        """
        self.dims = SimpleNamespace()
        self.lims = SimpleNamespace()

        self.dims.x = x_dim
        self.dims.y = x_dim if y_dim is None else y_dim
        self.dims.z = x_dim if z_dim is None else z_dim

        self.lims.x = x_lim
        self.lims.y = x_lim if y_lim is None else y_lim
        self.lims.z = x_lim if z_lim is None else z_lim

        self.N = self.dims.x * self.dims.y * self.dims.z  # number of voxles
        self.volume = (self.lims.x[1] - self.lims.x[0]) * (self.lims.y[1] - self.lims.y[0]) * (self.lims.z[1] - self.lims.z[0])  # the metric volume of the voxel presentation
        self.resolution = (self.volume / self.N) ** (1 / 3)  # the metric side length of each voxel

        self.values = np.zeros(shape=(self.dims.x, self.dims.y, self.dims.z), dtype=np.float32)  # store the value in the voxels
        self.weight = np.zeros_like(self.values)  # store the weight with corresponding voxels

    def __getitem__(self, pos):
        return self.values[pos]


def mesh_to_sdf(vertices, faces, face_normal=None, voxel_dim=256):
    # get the bounding box of the mesh
    # top_left = np.min(vertices, axis=0)
    # bottom_right = np.max(vertices, axis=0)

    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    face_normal = triangle_normal(v1, v2, v3) if face_normal is None else face_normal

    # sdf_class = voxels(x_dim=voxel_dim, x_lim=voxel_lim)

    for z in range(voxel_dim):
        for y in range(voxel_dim):
            for x in range(voxel_dim):
                print('[{},{},{}]'.format(x, y, z))
                points_3D = np.expand_dims(np.array([x, y, z]), axis=0)

                dist, _ = dist_points_to_triangles(v1, v2, v3, points_3D)
