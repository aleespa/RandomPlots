import gc
import random
import string

import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from colors.palettes import GrnGre
from common.technology import create_directory

FIGURE_NAME = 'lemniscate_curves'

def generate():
    create_directory(f"outputs/{FIGURE_NAME}")
    t_len = 200
    f_len = 50

    t = np.linspace(0, np.pi / 4, t_len)
    f = np.linspace(0, 10, f_len)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='#000000', dpi=400)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    x = (4 + f[:, None]) * np.sqrt(np.cos(2 * t)) * np.cos(t + f[:, None] / 2)
    y = (4 + f[:, None]) * np.sqrt(np.cos(2 * t)) * np.sin(t + f[:, None] / 2)

    for k in range(f_len):
        ax.plot(x[k, :], y[k, :], color=np.random.choice(GrnGre), lw=1)
        ax.plot(-x[k, :], y[k, :], color=np.random.choice(GrnGre), lw=1)
        ax.plot(-x[k, :], -y[k, :], color=np.random.choice(GrnGre), lw=1)
        ax.plot( x[k, :], -y[k, :], color=np.random.choice(GrnGre), lw=1)

    x = 4 * np.sqrt(np.cos(2 * t)) * np.cos(t)
    y = 4 * np.sqrt(np.cos(2 * t)) * np.sin(t)

    ax.plot(x, y, lw=7, color='#000000')
    ax.plot(x, -y, lw=7, color='#000000')
    ax.plot(-x, -y, lw=7, color='#000000')
    ax.plot(-x, y, lw=7, color='#000000')

    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    fig.savefig(f'outputs/{FIGURE_NAME}/{random_name}.png', facecolor='k')
    logger.info(f"{random_name}.png Saved")
    plt.close()
    gc.collect()
