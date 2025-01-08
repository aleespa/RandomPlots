import gc
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import create_directory, clear_folder, images_to_video


def generate():
    filename = sys.argv[1]
    n_frames = 300
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")
    n_grid = 10

    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
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
        x_1 = (1 - y**theta) ** (1 / theta)
        for j in range(n_grid - 15, n_grid + 5):
            for k in range(n_grid + 1):
                ax.plot(
                    (1 / n_grid) * x_1 + (k / n_grid),
                    (1 / n_grid) * y + (j / n_grid),
                    color='k',
                    lw=2.5,
                )
                ax.plot(
                    -(1 / n_grid) * x_1 + (k / n_grid),
                    (1 / n_grid) * y + (j / n_grid),
                    color='k',
                    lw=2.5,
                )
                ax.plot(
                    -(1 / n_grid) * x_1 + (k / n_grid),
                    -(1 / n_grid) * y + (j / n_grid),
                    color='k',
                    lw=2.5,
                )
                ax.plot(
                    (1 / n_grid) * x_1 + (k / n_grid),
                    -(1 / n_grid) * y + (j / n_grid),
                    color='k',
                    lw=2.5,
                )

        y1, y2 = 0, 1
        x1, x2 = 0, 1
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {theta:.8f} "
            f"frame {str(i + 1).zfill(3)}/{n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )
        gc.collect()

    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 60)
