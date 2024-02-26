import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import images_to_video, clear_folder, create_directory

os.add_dll_directory(r'C:\Users\Alejandro Lopez\Documents\codec')



def generate():
    def generate_plot(angle: float,
                      k: float,
                      color):

        circle = plt.Circle((
            np.cos(10 * angle) / (np.sin(10 * angle + k) + 2) * 10 * angle,
            np.sin(10 * angle) / (np.cos(10 * angle + k) + 2) * 10 * angle
                             ) , 0.2, color=color)
        ax.add_patch(circle)


    filename = sys.argv[1]
    n_frames = 900
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")

    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    angles = np.linspace(0, 9 * np.pi, 10000)
    for i, (theta) in enumerate(np.linspace(0, 3*np.pi, n_frames)):
        X, Y = ([np.cos(angle) / (np.sin(angle * theta) + 2) * angle/3 for angle in angles],
                [np.sin(angle) / (np.cos(angle * theta) + 2) * angle/3 for angle in angles])

        t1 = time.time()
        ax.clear()
        ax.plot(X,Y, color='k', lw=5)
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        y1, y2 = -10, 10
        x1, x2 = -10, 10
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(f"theta = {theta:.8f} frame {str(i + 1).zfill(3)}/{n_frames} time = {t2- t1:.2f} seconds")
        gc.collect()
    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 60)

