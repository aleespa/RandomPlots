import gc
import random
import string

import matplotlib.pylab as plt
import numpy as np
from loguru import logger
from matplotlib.collections import LineCollection

from common.technology import create_directory

FIGURE_NAME = 'polygons'

def generate(figure_size=(12, 12), dpi=100, seed=3123):
    create_directory(f"outputs/{FIGURE_NAME}")
    n = 20
    u = np.linspace(0, 2 * np.pi, n)
    fig, ax = plt.subplots(1, 1, figsize=figure_size, dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    n = len(u)
    x = np.cos(u)
    y = np.sin(u)
    i, j = np.triu_indices(n, k=1)
    num_lines = len(i)

    segments = np.stack(
        [np.column_stack((x[i], y[i])), np.column_stack((x[j], y[j]))], axis=1
    )
    cmap = plt.get_cmap("tab10")
    colors = [cmap(k % cmap.N) for k in range(num_lines)]

    line_collection = LineCollection(segments, linewidths=1.5, colors=colors)
    ax.add_collection(line_collection)

    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    fig.savefig(f'outputs/{FIGURE_NAME}/{random_name}.png', facecolor='k')
    logger.info(f"{random_name}.png Saved")
    plt.close()
    gc.collect()
