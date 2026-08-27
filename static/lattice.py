import gc

import matplotlib.pylab as plt
import numpy as np

from colors.palettes import TWLAGN, NEAURA
from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 100


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    for x in range(30):
        for y in range(30):
            p1 = np.array([x, y])
            r1 = rng.choice([-1, 0, 1])
            r2 = rng.choice([-1, 0, 1])
            ax.plot(
                [p1[0], p1[0] + r2],
                [p1[1], p1[1] + r1],
                color=rng.choice(TWLAGN),
            )
            ax.plot(
                [p1[0], p1[0] - r1],
                [p1[1], p1[1] - r2],
                color=rng.choice(TWLAGN),
            )

    settings.save_to_png(fig, 'k')
    plt.close()
    gc.collect()

    fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    for x in range(30):
        for y in range(30):
            p1 = np.array([x, y])
            for q in rng.choice([-1, 0, 1], 2, p=[0.2, 0.6, 0.2]):
                ax.plot(
                    [p1[0], p1[0]], [p1[1], p1[1] + q], color=rng.choice(NEAURA)
                )
            for q in rng.choice([-1, 0, 1], 2, p=[0.2, 0.6, 0.2]):
                ax.plot(
                    [p1[0], p1[0] + q], [p1[1], p1[1]], color=rng.choice(NEAURA)
                )
    settings.save_to_png(fig, 'k')
    plt.close()
    gc.collect()


if __name__ == '__main__':
    generate()
