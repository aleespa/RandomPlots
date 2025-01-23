from datetime import datetime
from math import sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from tools.image_processing import ImageProcessingSettings
from tools.technology import images_to_video


def brownian_bridge(n):
    T = np.linspace(0, 1, n)
    B = np.ones(n) * 0
    for i in range(n):
        xi = sqrt(2) * np.random.randn() / ((i + 1) * pi)
        B = B + xi * np.array([sin((i + 1) * pi * t) for t in T])
    return B


colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582']


def generate(settings=ImageProcessingSettings()):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor="#000000")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)
    ax.scatter([0, 1], [0, 0], s=40, color='#ff0000', zorder=200)
    for i in range(120):
        ax.plot(
            np.linspace(0, 1, 2000), brownian_bridge(2000),
            color=np.random.choice(colors), alpha=0.6
        )
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        plt.savefig(
            f'outputs/{settings.filename}/{time_string}.png',
            facecolor='#000000',
        )
    images_to_video(f'outputs/{settings.filename}',
                    f'{settings.filename}.mp4',
                    20)
