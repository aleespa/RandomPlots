import gc
import random
import string

import matplotlib.pylab as plt
import numpy as np
import toml
from loguru import logger

from tools.technology import create_directory

colors1 = ['#ccabd8', '#8474a1', '#6ec6ca', '#08979d', '#055b5c']
colors2 = ['#ff00b4', '#00ffbc', '#8ea5ff', '#ffffff', '#c493ff']


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='k')
    for x in range(30):
        for y in range(30):
            p1 = np.array([x, y])
            r1 = np.random.choice([-1, 0, 1])
            r2 = np.random.choice([-1, 0, 1])
            ax.plot(
                [p1[0], p1[0] + r2],
                [p1[1], p1[1] + r1],
                color=np.random.choice(colors1),
            )
            ax.plot(
                [p1[0], p1[0] - r1],
                [p1[1], p1[1] - r2],
                color=np.random.choice(colors1),
            )

    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    logger.info(f"{name}.png Saved")
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
    plt.close()
    gc.collect()

    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='k')
    for x in range(30):
        for y in range(30):
            p1 = np.array([x, y])
            for q in np.random.choice([-1, 0, 1], 2, p=[0.2, 0.6, 0.2]):
                ax.plot(
                    [p1[0], p1[0]], [p1[1], p1[1] + q], color=np.random.choice(colors2)
                )
            for q in np.random.choice([-1, 0, 1], 2, p=[0.2, 0.6, 0.2]):
                ax.plot(
                    [p1[0], p1[0] + q], [p1[1], p1[1]], color=np.random.choice(colors2)
                )
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    logger.info(f"{name}.png Saved")
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
    plt.close()
    gc.collect()
