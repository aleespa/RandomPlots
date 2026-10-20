import gc

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings

colors = [
    "#23895d",
    "#7fd677",
    "#eadb87",
    "#eeda9e",
    "#b8b8b8",
]
mirrors = [(1, 1), (-1, 1), (-1, -1), (1, -1)]


def lemniscate(size, turn, t):
    r = size * np.sqrt(np.cos(2 * t))
    return r * np.cos(t + turn), r * np.sin(t + turn)


def generate(settings: ImageProcessingSettings = None):
    """Fifty lemniscates, growing and turning, mirrored
    four ways around a black infinity sign."""
    settings = settings or ImageProcessingSettings(1)
    t = np.linspace(0, np.pi / 4, 200)

    fig = plt.figure(figsize=(12, 12), dpi=150, facecolor="k")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-10, 10), ylim=(-10, 10))
    ax.axis("off")
    for k in np.linspace(0, 10, 50):
        x, y = lemniscate(4 + k, k / 2, t)
        for a, b in mirrors:
            color = settings.rng.choice(colors)
            ax.plot(a * x, b * y, color=color, lw=1)
    x, y = lemniscate(4, 0, t)
    for a, b in mirrors:
        ax.plot(a * x, b * y, color="k", lw=7)
    settings.save_to_png(fig, "black")
    plt.close()
    gc.collect()


if __name__ == "__main__":
    generate()
