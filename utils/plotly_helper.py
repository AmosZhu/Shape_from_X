"""
Author: dizhong zhu
Date: 15/08/2022
"""

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from GeoUtils.common import make_homegenous


def get_camera(M=np.eye(4), scale=1):
    """
    Get camera by type
    :param M: transform the matrix in [3X4]
    :return:
    """
    vertices = np.array([[-0.5, -0.5, 1],
                         [0.5, -0.5, 1],
                         [0.5, 0.5, 1],
                         [-0.5, 0.5, 1],
                         [0, 0, 0]]) * scale
    faces = np.array([[0, 1, 2],
                      [0, 2, 3],
                      [0, 1, 4],
                      [1, 2, 4],
                      [2, 3, 4],
                      [3, 0, 4]])

    v_h = make_homegenous(vertices)
    vertices_T = v_h @ M.T

    vertices_T = vertices_T[:, :3] / vertices_T[:, 3:]

    wireframe = vertices_T[[0, 1, 2, 3, 0, 4, 1, 2, 4, 3], :]
    return vertices_T, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    w = wireframe
    # for w in wireframe.T:
    wireframe_merged[0] += [float(n) for n in w[:, 0]] + [None]
    wireframe_merged[1] += [float(n) for n in w[:, 1]] + [None]
    wireframe_merged[2] += [float(n) for n in w[:, 2]] + [None]
    return wireframe_merged


# damnly slow for many images
def plotly_imageset(images, nrow=0):
    n_view = images.shape[0]

    nrows = np.int(np.ceil(np.sqrt(n_view))) if nrow <= 0 else nrow
    ncols = np.int(np.ceil(n_view / nrows))

    h, w = images.shape[1:3]

    fig = make_subplots(rows=nrows, cols=ncols)

    for idx, img in enumerate(images):
        m = int(np.floor(idx / ncols))
        n = int(idx - m * ncols)
        fig.add_trace(go.Image(z=img), row=m + 1, col=n + 1)

    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


def plotly_pointcloud_and_camera(pt3D: list, cam_matrices=None):
    '''
    :param pt3D: list of point cloud in 3D spaces
    :param cam_matrices: camera matrices in size [N, 4 x 4]
    :return:
    '''
    plot_data = []

    for cam_P in cam_matrices:
        vertices, faces, wireframe = get_camera(M=cam_P, scale=0.2)
        plot_data.append(
            go.Mesh3d(
                x=vertices[:, 0].tolist(),
                y=vertices[:, 1].tolist(),
                z=vertices[:, 2].tolist(),
                i=faces[:, 0].tolist(),
                j=faces[:, 1].tolist(),
                k=faces[:, 2].tolist(),
                color='#00ff00',
                opacity=0.01,
            )
        )

        wireframe_merged = merge_wireframes(wireframe)
        plot_data.append(
            go.Scatter3d(
                x=wireframe_merged[0],
                y=wireframe_merged[1],
                z=wireframe_merged[2],
                mode="lines",
                line=dict(color="#ff0000", width=1),
                opacity=1,
            )
        )

    for pc in pt3D:
        plot_data.append(
            go.Scatter3d(
                x=pc[..., 0],
                y=pc[..., 1],
                z=pc[..., 2],
                mode='markers',
                marker=dict(size=[1] * pc.shape[0],
                            color=[0] * pc.shape[0])
            )
        )

    fig = go.Figure()

    for data in plot_data:
        fig.add_trace(data)

    fig.update_layout(showlegend=False)

    return fig
