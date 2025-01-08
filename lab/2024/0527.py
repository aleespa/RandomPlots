import gc
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from tools.technology import create_directory, clear_folder


def generate():
    filename = sys.argv[1]
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")

    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    n_frames = 10
    max_depth = 6
    draw_tree(ax, 0, 0, np.pi / 2, 0, 16, np.pi / 5)
    ax.set_ylim(190, 310)
    ax.set_xlim(-60, 60)
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
    t2 = time.time()
    logger.info(f"Finished")
    gc.collect()


def draw_tree(ax, x, y, angle, depth, max_depth, angle_shift):
    if depth > max_depth:
        return

    # Calculate the end of the branch
    branch_length = 100 * (0.7 ** depth)
    x_end = x + branch_length * np.cos(angle)
    y_end = y + branch_length * np.sin(angle)

    # Draw the branch
    ax.plot([x, x_end], [y, y_end], color='k')

    # Recursively draw the next branches
    new_depth = depth + 1
    draw_tree(ax, x_end, y_end,
              angle - angle_shift, new_depth, max_depth, angle_shift)
    draw_tree(ax, x_end, y_end,
              angle + angle_shift, new_depth, max_depth, angle_shift)
