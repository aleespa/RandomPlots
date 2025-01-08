import gc
import random
import string
from math import sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np
import toml
from loguru import logger

from tools.technology import create_directory


def browninan_bridge(n):
    T = np.linspace(0, 1, n)
    B = np.ones(n) * 0
    for i in range(n):
        xi = sqrt(2) * np.random.randn() / ((i + 1) * pi)
        B = B + xi * np.array([sin((i + 1) * pi * t) for t in T])
    return B


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")

    color1 = ['#96ceb4', '#ffeead', '#ff6f69', '#ffcc5c', '#88d8b0']
    color2 = ['#f7f4a3', '#7fccec', '#6a81d9', '#a479c9', '#dfdfdf']
    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='k')
    n = 500
    for k in range(35):
        X = browninan_bridge(n)
        ax.plot(
            np.linspace(0, 1, n),
            [X[i] + k * i / n - k for i in range(n)],
            color=np.random.choice(color1),
            lw=2,
            alpha=0.8,
        )
    for k in range(35):
        X = browninan_bridge(n)
        ax.plot(
            np.linspace(0, 1, n),
            [X[i] + (k - 34) * i / n - k for i in range(n)],
            color=np.random.choice(color2),
            lw=2,
            alpha=0.8,
        )
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    logger.info(f"{name}.png Saved")
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
    logger.info(f"Finished")
    gc.collect()
