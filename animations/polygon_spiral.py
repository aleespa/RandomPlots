from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from colors.palettes import ECOSPL
from common.simulation import get_rng
from common.technology import clear_folder, create_directory, images_to_video

FIGURE_NAME = 'polygon_spiral'


def generate(figure_size=(12, 12), dpi=100, seed=3123):
    create_directory(f"outputs/{FIGURE_NAME}")
    clear_folder(f"outputs/{FIGURE_NAME}")
    rng = get_rng(seed)

    fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')

    y1, y2 = -260, 260
    x1, x2 = -260, 260
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    n = 350
    def cubo(a, b, t):
        c_x = np.array([[a, -a], [b, -a], [b, -b], [a, -b], [a, -a]])
        c_y = np.array([[a, a], [a, b], [b, b], [b, a], [a, a]])
        trig_matrix = np.vstack((np.cos(t), np.sin(t)))
        x = (c_x @ trig_matrix).reshape(-1, )
        y = (c_y @ trig_matrix).reshape(-1, )

        ax.plot(x, y, color=rng.choice(ECOSPL), alpha=0.9, lw=2)

    output_frames_path = Path('outputs') / f'{FIGURE_NAME}' / "frames"
    create_directory(output_frames_path)
    clear_folder(output_frames_path)
    for i, t in enumerate(np.linspace(0, 20 * np.pi, n)[::-1]):
        cubo(1 * t, 3 * t, t)
        file_name = f'frame{str(i).zfill(4)}'
        plt.savefig(output_frames_path / f'{file_name}.png', facecolor='black')
        logger.info(f"{file_name}.png Saved")

    images_to_video(output_frames_path, f'{FIGURE_NAME}.mp4', 30)
    logger.success(f"Finished")
