import gc
import random
import string

import toml
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import create_directory, clear_folder


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")

    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1), facecolor='#f4f0e7')
    ...
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    logger.info(f"{name}.png Saved")
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
    logger.info(f"Finished")
    plt.close()
    gc.collect()