import gc

import matplotlib.pylab as plt
import numpy as np
from matplotlib.collections import LineCollection

from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 100


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 20
    u = np.linspace(0, 2 * np.pi, n)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE, dpi=DPI)
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

    settings.save_to_png(fig, 'k')
    plt.close()
    gc.collect()


if __name__ == '__main__':
    generate()
