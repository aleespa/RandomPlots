import gc
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from colors.palettes import SUNWAV
from common.technology import create_directory, clear_folder, images_to_video

FIGURE_NAME = 'rain'



def norm(x, t):
    return np.exp(-((x - t) ** 2) / 550) * (x < t)

def generate(figure_size=(12, 12), dpi=150, seed=None):
    output_frames_path = Path('outputs') / FIGURE_NAME / "frames"
    create_directory(output_frames_path)
    clear_folder(output_frames_path)
    rng = np.random.default_rng(seed)

    u1 = rng.uniform(0, 1, 200)
    u2 = rng.uniform(0, 1, 200)
    for t in range(200):
        fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        colors = [color for color, count in SUNWAV for _ in range(count)]
        ax.scatter(
            u1, u2, s=[2600 * norm(x, t) for x in range(200)], alpha=0.7, color=colors
        )
        file_name = f'frame{str(t).zfill(4)}.png'
        plt.savefig(output_frames_path / file_name, facecolor='black')
        logger.info(f"{file_name} Saved")
        plt.close()
        gc.collect()
    images_to_video(output_frames_path, f'{FIGURE_NAME}.mp4', 30)
    logger.success("Video created.")