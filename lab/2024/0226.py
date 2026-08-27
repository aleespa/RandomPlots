import time

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    n_frames = 900

    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    angles = np.linspace(0, 9 * np.pi, 10000)
    for i, (theta) in enumerate(np.linspace(0, 3 * np.pi, n_frames)):
        X, Y = (
            [
                np.cos(angle) / (np.sin(angle * theta) + 2) * angle / 3
                for angle in angles
            ],
            [
                np.sin(angle) / (np.cos(angle * theta) + 2) * angle / 3
                for angle in angles
            ],
        )

        t1 = time.time()
        ax.clear()
        ax.plot(X, Y, color='k', lw=5)
        y1, y2 = -10, 10
        x1, x2 = -10, 10
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)
        fig.savefig(settings.frames_path / f'frame{i:04d}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {theta:.8f} frame {str(i + 1).zfill(3)}/{n_frames} time = {t2- t1:.2f} seconds"
        )
    plt.close(fig)
    settings.save_video(60)


if __name__ == '__main__':
    generate()
