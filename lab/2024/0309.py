import gc
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import clear_folder, create_directory, images_to_video


def generate():
    filename = sys.argv[1]
    n_frames = 600
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")

    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    angles = np.linspace(-10, 10, 10000)
    n_waves = 75
    wavelengths = 25
    speed = np.random.uniform(0, 1, (wavelengths, 2, n_waves))
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, n_frames)):
        t1 = time.time()
        ax.clear()
        for k, h in enumerate(np.linspace(-15, 25, n_waves)):
            X = h + 0.4 * sum(
                speed[i, 1, k] * np.cos(angles * speed[i, 0, k] + theta)
                for i in range(wavelengths)
            ) * (
                0.5 * np.exp(-((angles + h - 10) ** 2))
                + 0.5 * np.exp(-((angles - h + 10) ** 2))
            )
            ax.plot(angles, X, color='k', lw=3)
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        y1, y2 = 2, 12
        x1, x2 = -10, 10
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {theta:.8f} "
            f"frame {str(i + 1).zfill(3)}/{n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )
        gc.collect()
    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 60)
