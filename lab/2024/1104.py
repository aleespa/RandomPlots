import gc
import random
import string
import subprocess
from datetime import datetime

import matplotlib.pylab as plt
import numpy as np
import matplotlib.colors as mcolors
import toml
from loguru import logger

from tools.technology import create_directory, images_to_video, clear_folder

colors = [ "#f4f0e7", "#000007",'#ffffff']  # Red, Green, Blue, Yellow, Cyan

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)

def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")


    logger.info(f"Starting calculation")
    for theta in np.linspace(0, 2 *np.pi , 600):
        z = 1.1 * np.exp(1j * theta) * ((2 - np.exp(1j * theta)) / 4)

        logger.info(f"theta = {theta}")
        subprocess.run(["java", "julia", f"{z.real}", f'{z.imag}', f'outputs/{filename}/temp.txt'],
                                capture_output=True, text=True)

        number_iterations = np.loadtxt(f"outputs/{filename}/temp.txt",
                                       delimiter=",", dtype=float)
        fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor='black')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(number_iterations, cmap=cmap, vmax=200, vmin=0)

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        logger.info(f"{time_string}.png Saved")
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        plt.close()
        gc.collect()
    images_to_video(f'outputs/{filename}',
                    f'{filename}.mp4', 60)
    logger.info(f"Finished")