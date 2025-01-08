import gc
import random
import string

import matplotlib.colors as mcolors
import numpy as np
import toml
from loguru import logger
from matplotlib import pyplot as plt

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
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")

    t = np.linspace(0, 2 * np.pi, 1000)
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor='#f4f0e7')
    random_ks = np.random.random_integers(1, 20, size=10)
    for k in range(48):
        plot_epicycloid(t, k, ax)
        logger.info(f"k = {k}")
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    logger.info(f"{name}.png Saved")
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
    logger.info(f"Finished")
    plt.close()
    gc.collect()


def epicycloid(t: np.array, k: float = 0.5):
    return (k + 1) * np.exp(1j * t) - np.exp(1j * (k + 1) * t)


def plot_epicycloid(t, k, ax):
    s = epicycloid(t, k)
    ax.plot(s.real, s.imag, lw=3.2, alpha=0.9, color=cmap(np.random.uniform()))
