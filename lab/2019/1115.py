from math import sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings

colors = ['#ff9b9b', '#f8ff90', '#a9ff8f', '#22ba5a', '#58c0e7']


def generate(settings=ImageProcessingSettings()):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor="#000000")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    for x in np.linspace(-1, 1, 121):
        ax.plot(
            [x, x * x], [0, sqrt(1 - x**2)], alpha=0.8, color=np.random.choice(colors)
        )
        ax.plot(
            [x, x * x], [0, -sqrt(1 - x**2)], alpha=0.8, color=np.random.choice(colors)
        )
        ax.plot(
            [x, x * x], [0, -sqrt(1 - x**2) * x], alpha=0.8, color=np.random.choice(colors)
        )
        ax.plot(
            [x, x * x], [0, sqrt(1 - x**2) * x], alpha=0.8, color=np.random.choice(colors)
        )
    plt.savefig(
        f'outputs/{settings.filename}/image_1.png', facecolor='black'
    )
