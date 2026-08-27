import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import njit

from common.image_processing import ImageProcessingSettings


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
    l_cx: np.array, l_cy: np.array, area: np.array, i: int, settings: ImageProcessingSettings
):
    start_color = '#f4f0e7'  # Light color you specified
    end_color = '#000000'  # Black

    # Create a colormap from the specified colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])

    h, _, _ = np.histogram2d(l_cx, l_cy, bins=4000, range=area)

    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    ax.imshow(np.log(h + 1), vmin=0, vmax=5, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    settings.save_numbered_frame(i, 'k')
    del l_cx, l_cy


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n_points = 600
    n_iter = 150
    area = np.array([[-0.5, 1], [-1, 1]])
    for i, b in enumerate(np.linspace(0, 2 * np.pi, 600)):
        a = 0.5
        l_cx, l_cy = calc_orbit(n_points, a, b, n_iter)
        generate_plot(l_cx, l_cy, area, i, settings)
    settings.save_video(30)


if __name__ == '__main__':
    generate()
