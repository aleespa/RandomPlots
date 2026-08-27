from math import sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings

colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582']


def brownian_bridge(n, rng):
    T = np.linspace(0, 1, n)
    B = np.ones(n) * 0
    for i in range(n):
        xi = sqrt(2) * rng.standard_normal() / ((i + 1) * pi)
        B = B + xi * np.array([sin((i + 1) * pi * t) for t in T])
    return B


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings()
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor="#000000")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)
    ax.scatter([0, 1], [0, 0], s=40, color='#ff0000', zorder=200)
    for i in range(120):
        ax.plot(
            np.linspace(0, 1, 2000), brownian_bridge(2000, settings.rng),
            color=settings.rng.choice(colors), alpha=0.6
        )
        fig.savefig(
            settings.frames_path / f'frame{i:04d}.png', facecolor='#000000'
        )
    plt.close()
    settings.save_video(20)


if __name__ == '__main__':
    generate()
