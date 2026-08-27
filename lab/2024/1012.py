import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    fig, _ = plt.subplots(figsize=(12, 12), dpi=250)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    t1 = np.linspace(0, 2 * np.pi, 250)
    t2 = np.linspace(0, 2 * np.pi, 25)
    for s in np.linspace(0, np.pi, 20):
        ax.scatter(
            np.cos(t2) * (1 + s * np.cos(t2)),
            np.sin(t2) * (1 + s * np.cos(t2)),
            color='k',
            lw=0,
            s=30,
        )
        ax.plot(
            np.cos(t1) * (1 + s * np.cos(t1)),
            np.sin(t1) * (1 + s * np.cos(t1)),
            color='k',
            lw=0.50,
        )
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-3, 3)
    settings.save_to_png(fig, 'k')
    logger.info(f"Finished")


if __name__ == '__main__':
    generate()
