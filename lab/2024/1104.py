import gc
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
import toml
from loguru import logger

from tools.technology import create_directory, images_to_video

colors = ["#f4f0e7", "#000007", "#EB251F", '#ffffff']  # Red, Green, Blue, Yellow, Cyan

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    # clear_folder(f"outputs/{filename}")

    logger.info(f"Starting calculation")
    n = 600
    theta = np.linspace(0, 2 * np.pi, n)
    # create_directory(f"outputs/{filename}/temp")
    # clear_folder(f"outputs/{filename}/temp")
    # z = 1.1 * np.exp(1j * theta) * ((2 - np.exp(1j * theta)) / 4)
    # for i in range(n):
    #
    #     subprocess.run(["java", "julia", f"{z[i].real}", f'{z[i].imag}',
    #                     f'outputs/{filename}/temp/{i}.txt'],
    #                    capture_output=True, text=True)
    #
    #     logger.info(f"Calculation finished for theta = {theta[i]}")

    for i in range(n):
        number_iterations_1 = np.loadtxt(
            f"outputs/{filename}/temp/{i}.txt", delimiter=",", dtype=float
        )
        number_iterations_2 = np.loadtxt(
            f"outputs/{filename}/temp/{n-i-1}.txt", delimiter=",", dtype=float
        )
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#f4f0e7")
        ax.set_xticks([])
        ax.set_yticks([])

        number_iterations = number_iterations_1 + number_iterations_2
        ax.imshow(number_iterations[500:3500, 500:3500], cmap=cmap, vmax=200, vmin=3)

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        logger.info(f"{time_string}.png Saved")
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor="#f4f0e7")
        plt.close()
        gc.collect()
    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 30)
    logger.info(f"Finished")
