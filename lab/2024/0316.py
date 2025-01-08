import gc
import time
from datetime import datetime

import numpy as np
import toml
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import create_directory, clear_folder, images_to_video


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    n_frames = 600
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")
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

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {t:.8f} "
            f"frame {str(i + 1).zfill(3)}/{2 * n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )
        gc.collect()
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

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {t:.8f} "
            f"frame {str(n_frames + i + 1).zfill(3)}/{2 * n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )
        gc.collect()

    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 60)
