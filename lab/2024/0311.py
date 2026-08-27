import time

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings
from common.simulation import brownian_bridge


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n_frames = 30

    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    y = np.linspace(0, 1, 100)
    t = np.concatenate(
        (
            np.linspace(0.001, 1, int(n_frames / 2)),
            1 / np.linspace(0.001, 1, int(n_frames / 2))[::-1],
        )
    )
    for i, theta in enumerate(t):
        ax.clear()
        t1 = time.time()
        n_steps = (i + 1) * 10
        t = np.linspace(0, 1, n_steps)
        for k in range(10):
            y = brownian_bridge(n_steps, settings.rng)
            ax.plot(t + k, y * (1 + k**0.5), color='k')
        y1, y2 = -10, 10
        x1, x2 = 0, 10
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)

        fig.savefig(settings.frames_path / f'frame{i:04d}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {theta:.8f} "
            f"frame {str(i + 1).zfill(3)}/{n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )

    plt.close(fig)
    settings.save_video(6)


if __name__ == '__main__':
    generate()
