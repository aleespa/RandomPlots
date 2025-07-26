import gc
import random
import string

import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from colors.palettes import GYRYG, YBBPG
from common.simulation import brownian_bridge, get_rng
from common.technology import create_directory

FIGURE_NAME = 'bb_connections'

def generate(figure_size=(12, 12), dpi=100, seed=None):
    create_directory(f"outputs/{FIGURE_NAME}")
    rng = get_rng(seed)
    fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    n = 500
    for k in range(35):
        x = brownian_bridge(n, rng)
        ax.plot(
            np.linspace(0, 1, n),
            [x[i] + k * i / n - k for i in range(n)],
            color=rng.choice(GYRYG),
            lw=2,
            alpha=0.8,
        )
    for k in range(35):
        x = brownian_bridge(n, rng)
        ax.plot(
            np.linspace(0, 1, n),
            [x[i] + (k - 34) * i / n - k for i in range(n)],
            color=rng.choice(YBBPG),
            lw=2,
            alpha=0.8,
        )
    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    fig.savefig(f'outputs/{FIGURE_NAME}/{random_name}.png', facecolor='k')
    logger.info(f"{random_name}.png Saved")
    plt.close()
    gc.collect()
