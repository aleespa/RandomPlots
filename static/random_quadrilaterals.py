import gc
import random
import string

import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from common.technology import create_directory

FIGURE_NAME = 'random_quadrilaterals'

colors = ['#3a3663', '#414977', '#476589', '#4c7c9a', '#58c0e7']

def generate(figure_size=(12, 12), dpi=150, seed=None):
    create_directory(f"outputs/{FIGURE_NAME}")
    rng = np.random.default_rng(seed)

    for fig_num in [1, 2]:
        fig, _ = plt.subplots(figsize=figure_size, dpi=dpi)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
        for _ in range(100):
            R = rng.uniform(-5, 5, 4)
            if fig_num == 1:
                x = [0, -R[0], -R[1], 0]
                y = [0, -R[2], R[3], 0]
            else:
                x = [0, -R[0], -R[0], 0]
                y = [0, -R[1], R[1], 0]
            ax.fill_between(x, y, alpha=0.25, color=rng.choice(colors))
            ax.plot(x, y, alpha=0.2, color='white', lw=1)
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        fig.savefig(f'outputs/{FIGURE_NAME}/{random_name}.png', facecolor='black')
        logger.info(f"{random_name}.png Saved")
        plt.close()
        gc.collect()