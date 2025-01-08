import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import images_to_video

os.add_dll_directory(r'C:\Users\Alejandro Lopez\Documents\codec')


def generate():
    def generate_plot(angle: float):
        ax.clear()  # Clear previous plot data
        for k in range(15):
            circle = plt.Circle(
                (
                    k * np.sin(angle * k) * np.sin(angle) * np.cos(angle),
                    k * np.cos(angle) * np.sin(angle),
                ),
                0.35,
                color='k',
            )
            ax.add_patch(circle)

        y1, y2 = -8, 8
        x1, x2 = -8, 8
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)

    filename = sys.argv[1]
    n_frames = 780

    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    for i, (theta) in enumerate(np.linspace(0, 2 * np.pi, n_frames)):
        t1 = time.time()
        generate_plot(theta)
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {theta:.8f} frame {str(i + 1).zfill(3)}/{n_frames} time = {t2- t1:.2f} seconds"
        )
        gc.collect()
    images_to_video(f'outputs/{filename}', '20240224.mp4', 60)
