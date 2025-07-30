import gc
from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from colors.palettes import RedWht
from common.image_processing import clear_folder
from common.simulation import get_rng
from common.technology import create_directory, images_to_video

FIGURE_NAME = 'sliding_orbits'


def generate(figure_size=(12, 12), dpi=100, seed=3123):
    create_directory(f"outputs/{FIGURE_NAME}")
    clear_folder(f"outputs/{FIGURE_NAME}")
    rng = get_rng(seed)
    t = rng.uniform(0, 2 * np.pi, 580)
    color_list = rng.choice(RedWht, 580)
    for i, s in enumerate(np.linspace(0, 2, 180)):
        frame(s, t, color_list, figure_size, dpi)

        file_name = f'frame{str(i).zfill(4)}'
        plt.savefig(f'outputs/{FIGURE_NAME}/{file_name}.png', facecolor='black')
        logger.info(f"{file_name}.png Saved")
        plt.close()
        gc.collect()

    images_to_video(Path(f"outputs/{FIGURE_NAME}"), f'{FIGURE_NAME}.mp4', 30)
    logger.info(f"Finished")


def frame(s, t: np.ndarray, color_list: list, figure_size=(12, 12), dpi=100):

    fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    r_line = np.linspace(0, 0.5, 50)
    r_sin = np.linspace(0, s, 50)

    for m in range(1, 10):
        for k in range(m * 6 + 1):
            idx = k * m
            if idx >= len(t):
                continue  # Avoid index overflow

            angle_line = t[idx] + r_line
            angle_sin = t[idx] + r_sin

            x = m * np.cos(angle_line)
            y = m * np.sin(angle_sin)

            ax.plot(x, y, color=color_list[idx], alpha=0.8, lw=2)
            ax.scatter(
                [m * np.cos(t[idx])],
                [m * np.sin(t[idx])],
                s=22,
                zorder=10,
                color='#00b0b0',
                alpha=0.8,
            )
