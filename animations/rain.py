import matplotlib.pylab as plt
import numpy as np

from colors.palettes import SUNWAV
from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 150


def norm(x, t):
    return np.exp(-((x - t) ** 2) / 550) * (x < t)


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    u1 = rng.uniform(0, 1, 200)
    u2 = rng.uniform(0, 1, 200)
    for t in range(200):
        fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        colors = [color for color, count in SUNWAV for _ in range(count)]
        ax.scatter(
            u1, u2, s=[2600 * norm(x, t) for x in range(200)], alpha=0.7, color=colors
        )
        settings.save_numbered_frame(t, 'black')
    settings.save_video(30)


if __name__ == '__main__':
    generate()
