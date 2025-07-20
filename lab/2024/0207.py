import gc
import sys
from datetime import datetime

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import njit

from common.technology import images_to_video

matplotlib.use('Agg')
import os

# Add directory containing OpenH264 DLL to the DLL search path
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
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])

    h, _, _ = np.histogram2d(l_cx, l_cy, bins=4000, range=area)

    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    ax.imshow(np.log(h + 1), vmin=0, vmax=5, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
    plt.close()
    del l_cx, l_cy
    gc.collect()


def generate():
    filename = sys.argv[1]
    n_points = 600
    n_iter = 150
    # a, b = 5.45, 4.55
    for a, b in np.random.uniform(0, 2 * np.pi, (0, 2)):
        l_cx, l_cy = calc_orbit(n_points, a, b, n_iter)
        area = np.array([[-1, 1], [-1, 1]])
        generate_plot(l_cx, l_cy, area, filename, f'{a:.5f}{b:.5f}')

    # n_points = 600
    # n_iter = 100
    # a, b = 5.46781, 5.13645
    # l_cx, l_cy = calc_orbit(n_points, a, b, n_iter)
    # area = np.array([[-1, 0.2], [-1, 1]])
    # generate_plot(l_cx, l_cy, area, filename, f'version_1')

    # n_points = 600
    # n_iter = 200
    # a, b = 3.19899, 0.79022
    # l_cx, l_cy = calc_orbit(n_points, a, b, n_iter)
    # area = np.array([[-0.6, 0.9], [-0.4, 1]])
    # generate_plot(l_cx, l_cy, area, filename, f'version_2')

    # n_points = 600
    # n_iter = 200
    # a, b = 0.44717, 1.98510
    # l_cx, l_cy = calc_orbit(n_points, a, b, n_iter)
    # area = np.array([[-0.55, 1], [-1, 0.9]])
    # generate_plot(l_cx, l_cy, area, filename, f'version_3')

    # for b in np.linspace(0,2*np.pi, 600):
    #     a = 0.5
    #     l_cx, l_cy = calc_orbit(n_points, a, b, n_iter)
    #     area = np.array([[-0.5, 1], [-1, 1]])
    #     generate_plot(l_cx, l_cy ,area, 'animation2',f'{a:.5f}{b:.5f}')
    #     del l_cx, l_cy
    #     gc.collect()

    images_to_video(f'outputs/animation2', 'non_linear_final.mp4', 30)
