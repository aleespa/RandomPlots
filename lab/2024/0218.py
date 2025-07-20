import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import njit

from common.technology import images_to_video

openh264_dir = r'C:\Users\Alejandro Lopez\Documents\codec'
os.add_dll_directory(openh264_dir)


@njit
def meshgrid(x, y):
    """
    This function replace np.meshgrid that is not supported by numba
    """
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j, k] = k  # change to x[k] if indexing xy
            yy[j, k] = j  # change to y[j] if indexing xy
    return xx, yy


@njit
def calc_orbit(n_points, a, b, n_iter):
    """
    This function calculate orbits in a vectorized fashion.

    -n_points: lattice of initial conditions, n_points x n_points in [-1,1]x[-1,1]
    -a: first parameter of the dynamical system
    -b: second parameter of the dynamical system
    -n_iter: number of iterations

    Return: two ndarrays: x and y coordinates of every point of every orbit.
    """
    area = [[-1, 1], [-1, 1]]
    x = np.linspace(area[0][0], area[0][1], n_points)
    y = np.linspace(area[1][0], area[1][1], n_points)
    xx, yy = meshgrid(x, y)
    l_cx, l_cy = np.zeros(n_iter * n_points**2), np.zeros(n_iter * n_points**2)
    for i in range(n_iter):
        xx_new = np.sin(xx**2 - yy**2 + a)
        yy_new = np.cos(2 * xx * yy + b)
        xx = xx_new
        yy = yy_new
        l_cx[i * n_points**2 : (i + 1) * n_points**2] = xx.flatten()
        l_cy[i * n_points**2 : (i + 1) * n_points**2] = yy.flatten()
    return l_cx, l_cy


def generate_plot(
    l_cx: np.array, l_cy: np.array, area: np.array, filename: str, name: str
):
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    start_color = '#f4f0e7'  # Light color you specified
    end_color = '#000000'  # Black

    # Create a colormap from the specified colors
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [start_color, "#902c2c", "#234349", end_color]
    )

    h, _, _ = np.histogram2d(l_cx, l_cy, bins=4000, range=area)
    figure_aspect_ratio = 9 / 16
    fig, _ = plt.subplots(figsize=(9, 16), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    extent = [-1, 1, -1, 1]
    ax.imshow(np.log(h + 1), vmin=0, vmax=5, cmap=cmap, extent=extent)
    y1, y2 = -1, 1
    x1, x2 = -1, 1
    w = x2 - x1
    h = y2 - y1
    z = (16 / 18) * w - (1 / 2) * h
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1 - z, y2 + z)
    plt.xticks([]), plt.yticks([])
    fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
    plt.close()
    del l_cx, l_cy
    gc.collect()


def generate():
    filename = sys.argv[1]
    n_points = 500
    n_iter = 200
    n_frames = 450

    for i, (theta) in enumerate(np.linspace(0, 2 * np.pi, n_frames)):
        t1 = time.time()
        l_cx, l_cy = calc_orbit(n_points, theta, np.pi / 2, n_iter)
        area = np.array([[-1, 1], [-1, 1]])
        generate_plot(l_cx, l_cy, area, 'tests', f'')
        t2 = time.time()
        logger.info(
            f"theta = {theta:.8f} frame {str(i + 1).zfill(3)}/{n_frames} time = {t2- t1:.2f} seconds"
        )
    images_to_video(f'outputs/tests', '20240218_V2.mp4', 30)
