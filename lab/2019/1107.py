import gc
import random
import string
from math import sqrt, cos, sin, pi

import matplotlib.pylab as plt
import numpy as np
import toml

from tools.technology import create_directory


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")

    T = np.linspace(0, pi / 4, 200)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='black', dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='k')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    for k in np.linspace(0, 10, 50):
        X, Y = np.array(
            [(4 + k) * sqrt(cos(2 * t)) * cos(t + k / 2) for t in T]
        ), np.array([(4 + k) * sqrt(cos(2 * t)) * sin(t + k / 2) for t in T])
        ax.plot(
            X,
            Y,
            color=np.random.choice(
                ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']
            ),
            lw=1,
        )
        ax.plot(
            -X,
            Y,
            color=np.random.choice(
                ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']
            ),
            lw=1,
        )
        ax.plot(
            -X,
            -Y,
            color=np.random.choice(
                ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']
            ),
            lw=1,
        )
        ax.plot(
            X,
            -Y,
            color=np.random.choice(
                ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']
            ),
            lw=1,
        )
    X, Y = np.array([4 * sqrt(cos(2 * t)) * cos(t) for t in T]), np.array(
        [4 * sqrt(cos(2 * t)) * sin(t) for t in T]
    )
    ax.plot(X, Y, lw=7, color=(0, 0, 0))
    ax.plot(X, -Y, lw=7, color=(0, 0, 0))
    ax.plot(-X, -Y, lw=7, color=(0, 0, 0))
    ax.plot(-X, Y, lw=7, color=(0, 0, 0))
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')
    plt.close()
    gc.collect()
