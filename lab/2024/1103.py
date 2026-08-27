import matplotlib.colors as mcolors
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings

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


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    t = np.linspace(0, 2 * np.pi, 1000)
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor='#f4f0e7')
    for k in range(48):
        plot_epicycloid(t, k, ax, settings.rng)
        logger.info(f"k = {k}")
    settings.save_to_png(fig, 'k')
    logger.info(f"Finished")
    plt.close()


def epicycloid(t: np.array, k: float = 0.5):
    return (k + 1) * np.exp(1j * t) - np.exp(1j * (k + 1) * t)


def plot_epicycloid(t, k, ax, rng):
    s = epicycloid(t, k)
    ax.plot(s.real, s.imag, lw=3.2, alpha=0.9, color=cmap(rng.uniform()))


if __name__ == '__main__':
    generate()
