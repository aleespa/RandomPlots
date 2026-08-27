import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from common.fractal import julia_set_v2
from common.image_processing import ImageProcessingSettings
from common.technology import create_directory

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


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    logger.info(f"Starting calculation")
    n = 10
    theta = np.linspace(0, 2 * np.pi, n)
    temp_dir = settings.output_path / settings.filename / 'temp'
    create_directory(temp_dir)
    z = 1.01 * np.exp(1j * theta) * ((2 - np.exp(1j * theta)) / 4)
    for i, s in enumerate(np.linspace(1, 1e-10, n)):
        julia_set_v2(z[i], 480, 0, 0, 1.5, str(temp_dir / f'{i}.txt'))
        logger.info(f"Calculation finished for s = {s}")
    for i in range(n):
        number_iterations = np.loadtxt(temp_dir / f"{i}.txt", dtype=float)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#f4f0e7")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(number_iterations, cmap=cmap)

        settings.save_numbered_frame(i, "#f4f0e7")
    settings.save_video(30)
    logger.info(f"Finished")


if __name__ == '__main__':
    generate()
