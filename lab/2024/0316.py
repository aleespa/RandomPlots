import time

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n_frames = 600
    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    y = np.linspace(0, 1, 100)
    angles = np.linspace(0, 2 * np.pi, n_frames)
    for i, t in enumerate(angles):
        ax.clear()
        t1 = time.time()
        for s in np.linspace(0, 2 * np.pi, 64):
            ax.plot(
                [
                    t * np.cos(s),
                    t * np.cos(t + s - (1 - t / (2 * np.pi)) * (np.pi / 2)),
                ],
                [t * np.sin(s), t * np.sin(t + s)],
                color='k',
            )
        y1, y2 = -7, 7
        x1, x2 = -7, 7
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)

        fig.savefig(settings.frames_path / f'frame{i:04d}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {t:.8f} "
            f"frame {str(i + 1).zfill(3)}/{2 * n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )
    for i, t in enumerate(angles[::-1]):
        ax.clear()
        t1 = time.time()
        for s in np.linspace(0, 2 * np.pi, 64):
            ax.plot(
                [t * np.cos(s), t * np.cos(t + s)],
                [t * np.sin(s), t * np.sin(t + s)],
                color='k',
            )
        y1, y2 = -7, 7
        x1, x2 = -7, 7
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)

        fig.savefig(settings.frames_path / f'frame{n_frames + i:04d}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {t:.8f} "
            f"frame {str(n_frames + i + 1).zfill(3)}/{2 * n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )

    plt.close(fig)
    settings.save_video(60)


if __name__ == '__main__':
    generate()
