import gc
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
import toml
from loguru import logger

from tools.fractal import julia_java, julia_set_v2
from tools.technology import create_directory, images_to_video, clear_folder

colors = [
    "#f4f0e7",
    "#e74c3c",  # Vibrant red
    "#f39c12",  # Vibrant orange
    "#27ae60",  # Vibrant green
    "#2980b9",  # Vibrant blue
    "#8e44ad",  # Vibrant purple
    '#000000',
]  # Red, Green, Blue, Yellow, Cyan

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")
    logger.info(f"Starting calculation")
    n = 10
    theta = np.linspace(0, 2 * np.pi, n)
    create_directory(f"outputs/{filename}/temp")
    clear_folder(f"outputs/{filename}/temp")
    z = 1.01 * np.exp(1j * theta) * ((2 - np.exp(1j * theta)) / 4)
    for i, s in enumerate(np.linspace(1, 1e-10, n)):

        julia_java(z[i], 480, 0, 0, 1.5, f'outputs/{filename}/temp/{i}.txt')
        logger.info(f"Calculation finished for s = {s}")
        julia_set_v2(z[i], 480, 0, 0, 1.5, f'outputs/{filename}/temp/{i}_.txt')
    for i in range(n):
        number_iterations = np.loadtxt(
            f"outputs/{filename}/temp/{i}.txt", delimiter=",", dtype=float
        )
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#f4f0e7")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(number_iterations, cmap=cmap)

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        logger.info(f"{time_string}.png Saved")

        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor="#f4f0e7")
        plt.close()
        gc.collect()
    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 30)
    logger.info(f"Finished")
