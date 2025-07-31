import gc
import random
import string

import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from colors.palettes import TWLAGN, NEAURA
from common.simulation import get_rng
from common.technology import create_directory, clear_folder

FIGURE_NAME = 'lattice'

def generate(figure_size=(12, 12), dpi=100, seed=None):
    create_directory(f"outputs/{FIGURE_NAME}")
    clear_folder(f"outputs/{FIGURE_NAME}")
    rng = get_rng(seed)

    fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    for x in range(30):
        for y in range(30):
            p1 = np.array([x, y])
            r1 = rng.choice([-1, 0, 1])
            r2 = rng.choice([-1, 0, 1])
            ax.plot(
                [p1[0], p1[0] + r2],
                [p1[1], p1[1] + r1],
                color=rng.choice(TWLAGN),
            )
            ax.plot(
                [p1[0], p1[0] - r1],
                [p1[1], p1[1] - r2],
                color=rng.choice(TWLAGN),
            )

    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    fig.savefig(f'outputs/{FIGURE_NAME}/{random_name}.png', facecolor='k')
    logger.success(f"{random_name}.png Saved")
    plt.close()
    gc.collect()

    fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    for x in range(30):
        for y in range(30):
            p1 = np.array([x, y])
            for q in rng.choice([-1, 0, 1], 2, p=[0.2, 0.6, 0.2]):
                ax.plot(
                    [p1[0], p1[0]], [p1[1], p1[1] + q], color=rng.choice(NEAURA)
                )
            for q in rng.choice([-1, 0, 1], 2, p=[0.2, 0.6, 0.2]):
                ax.plot(
                    [p1[0], p1[0] + q], [p1[1], p1[1]], color=rng.choice(NEAURA)
                )
    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    fig.savefig(f'outputs/{FIGURE_NAME}/{random_name}.png', facecolor='k')
    logger.success(f"{random_name}.png Saved")
    plt.close()
    gc.collect()
