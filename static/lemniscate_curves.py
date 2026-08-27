import gc

import matplotlib.pylab as plt
import numpy as np

from colors.palettes import GrnGre
from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 100


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    t_len = 200
    f_len = 50

    t = np.linspace(0, np.pi / 4, t_len)
    f = np.linspace(0, 10, f_len)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE, facecolor='#000000', dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    x = (4 + f[:, None]) * np.sqrt(np.cos(2 * t)) * np.cos(t + f[:, None] / 2)
    y = (4 + f[:, None]) * np.sqrt(np.cos(2 * t)) * np.sin(t + f[:, None] / 2)

    for k in range(f_len):
        for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
            ax.plot(sx * x[k, :], sy * y[k, :], color=rng.choice(GrnGre), lw=1)

    x = 4 * np.sqrt(np.cos(2 * t)) * np.cos(t)
    y = 4 * np.sqrt(np.cos(2 * t)) * np.sin(t)

    for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        ax.plot(sx * x, sy * y, lw=7, color='#000000')

    settings.save_to_png(fig, 'k')
    plt.close()
    gc.collect()


if __name__ == '__main__':
    generate()
