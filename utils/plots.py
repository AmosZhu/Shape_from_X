"""
Author: dizhong zhu
Date: 11/08/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_realsize(img):
    plt.figure(figsize=(int(img.shape[1] / 100), int(img.shape[0] / 100)))


def plot_landmarks(images, ld1, ld2, titles=None):
    n_view = images.shape[0]
    ncols = np.int(np.ceil(np.sqrt(n_view)))
    nrows = np.int(np.ceil(n_view / ncols))

    titles = [''] * n_view if titles is None else titles

    # plot openpose landmarks
    h, w = images.shape[1:3]
    sel = range(ld1.shape[1])
    fig, axarr = plt.subplots(nrows, ncols, figsize=(w * ncols / 100, h * nrows / 100))
    canvas = FigureCanvas(fig)

    axarr = np.array(axarr) if nrows * ncols == 1 else axarr
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))
    for ax, im, op_ld, J_ld, title in zip(axarr.ravel(), images, ld1, ld2, titles):
        ax.imshow(im)
        ax.plot(op_ld[sel, 0], op_ld[sel, 1], 'rx')
        ax.plot(J_ld[sel, 0], J_ld[sel, 1], 'g+')
        ax.axis('off')
        ax.set_title(title)

    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, dtype='uint8').reshape(height, width, 4)[..., :3]
    plt.show()

    return image


def plot_imageset(images, nrow=0, titles=None):
    n_view = images.shape[0]

    nrows = np.int(np.ceil(np.sqrt(n_view))) if nrow <= 0 else nrow
    ncols = np.int(np.ceil(n_view / nrows))

    titles = [''] * n_view if titles is None else titles

    # plot openpose landmarks
    h, w = images.shape[1:3]
    fig, axarr = plt.subplots(nrows, ncols, figsize=(w * ncols / 100, h * nrows / 100))
    canvas = FigureCanvas(fig)

    axarr = np.array(axarr) if nrows * ncols == 1 else axarr
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))
    for ax, im, title in zip(axarr.ravel(), images, titles):
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(title)

    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, dtype='uint8').reshape(height, width, 4)[..., :3]
    plt.show()

    return image
