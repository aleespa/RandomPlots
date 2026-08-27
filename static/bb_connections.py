import gc

import matplotlib.pylab as plt
import numpy as np

from colors.palettes import GYRYG, YBBPG
from common.image_processing import ImageProcessingSettings
from common.simulation import brownian_bridge

FIGURE_SIZE = (12, 12)
DPI = 100


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    n = 500
    for k in range(35):
        x = brownian_bridge(n, rng)
        ax.plot(
            np.linspace(0, 1, n),
            [x[i] + k * i / n - k for i in range(n)],
            color=rng.choice(GYRYG),
            lw=2,
            alpha=0.8,
        )
    for k in range(35):
        x = brownian_bridge(n, rng)
        ax.plot(
            np.linspace(0, 1, n),
            [x[i] + (k - 34) * i / n - k for i in range(n)],
            color=rng.choice(YBBPG),
            lw=2,
            alpha=0.8,
        )
    settings.save_to_png(fig, 'k')
    plt.close()
    gc.collect()


if __name__ == '__main__':
    generate()
