import gc
from datetime import datetime

import matplotlib.pylab as plt
import numpy as np
import matplotlib.colors as mcolors
import toml
from colorama.ansi import set_title
from loguru import logger

from tools.fractal import julia_java, julia_set_v2
from tools.settings import Settings
from tools.technology import create_directory, images_to_video, clear_folder

colors = [
    "#f4f0e7",
    "#e74c3c",  # Vibrant red
    "#f39c12",  # Vibrant orange
    "#27ae60",  # Vibrant green
    "#2980b9",  # Vibrant blue
    "#8e44ad",  # Vibrant purple
    '#000000']  # Red, Green, Blue, Yellow, Cyan

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)


def generate():
    settings = Settings()
    filename = settings.file_to_run
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")
    logger.info(f"Starting calculation")

    n = 50
    x = np.random.uniform(-1, 1, (n, n)) + np.random.uniform(-1, 1, (n, n)) * 1j

    for theta in np.linspace(0, 2 * np.pi, 300):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#f4f0e7")
        ax.set_xticks([])
        ax.set_yticks([])

        U = np.eye(n, dtype=complex)

        # Apply a 2D rotation to the first two dimensions
        U[:2, :2] = np.array([
            [np.cos(theta), - 1j * np.sin(theta)],
            [np.cos(-theta),  - 1j * np.sin(-theta)]
        ])

        # Rotate the matrix x using the unitary matrix U
        x_rotated = U @ x @ U.conj().T

        eigenvalues, eigenvectors = np.linalg.eig(x_rotated)
        norms = np.abs(eigenvalues)
        ax.scatter(eigenvalues.real, eigenvalues.imag,
                   color=[cmap(1 - z / max(norms)) for z in norms],
                   s=300, lw=0)
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        logger.info(f"{time_string}.png Saved")
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor="#f4f0e7")
        plt.close()
        gc.collect()
    images_to_video(f'outputs/{filename}',
                    f'{filename}.mp4', 60)
    logger.info(f"Finished")
