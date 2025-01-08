import gc
import random
import string

import matplotlib.colors
import matplotlib.pylab as plt
import numpy as np
import toml
from loguru import logger

from tools.technology import create_directory

colors = [
    "#3b3b3b",  # Dark gray
    "#4e5d6c",  # Muted dark blue
    "#4a4d73",  # Muted dark purple
    "#36454f",  # Dark slate gray
    "#555555",  # Medium gray
    "#2c3e50",  # Muted deep teal
    "#34495e",  # Muted navy blue
    "#e74c3c",  # Vibrant red
    "#f39c12",  # Vibrant orange
    "#27ae60",  # Vibrant green
    "#2980b9",  # Vibrant blue
    "#8e44ad",  # Vibrant purple
]
cmap = matplotlib.colors.ListedColormap(colors)


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")

    t = np.linspace(0, 2 * np.pi, 10000)
    for _ in range(50):
        fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
        for _ in range(4):
            a, b, c, d = sorted(np.random.randint(1, 24, 4))
            k, l = a / b, c / d
            plot_spiro(t, k, l, ax)
            logger.info(f"{a}/{b}, {c}/{d}")

        name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        logger.info(f"{name}.png Saved")
        fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
        plt.close()
        gc.collect()
    logger.info(f"Finished")


def spiro(t: np.array, k: float = 0.5, l: float = 0.5):
    return (1 - k) * np.exp(1j * t) + k * l * np.exp(-1j * t * (1 - k) / k)


def plot_spiro(t, k, l, ax):
    s = spiro(100 * t, k, l)
    ax.plot(s.real, s.imag, lw=3, alpha=0.9, color=cmap(np.random.uniform()))
