import gc
import sys

from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import create_directory, clear_folder


def generate():
    filename = sys.argv[1]
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")

    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    ...
    fig.savefig(f'outputs/{filename}/figure.png', facecolor='k')
    logger.info(f"Finished")
    gc.collect()