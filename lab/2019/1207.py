from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def BB(n, rng):
    T = np.linspace(0, 1, n)
    B = np.ones(n) * 0
    for i in range(n):
        xi = sqrt(2) * rng.standard_normal() / ((i + 1) * pi)
        B = B + xi * np.array([sin((i + 1) * pi * t) for t in T])
    return B


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    color1 = ['#96ceb4', '#ffeead', '#ff6f69', '#ffcc5c', '#88d8b0']
    color2 = ['#f7f4a3', '#7fccec', '#6a81d9', '#a479c9', '#dfdfdf']
    fig = plt.figure(figsize=(14, 14), facecolor=(0, 0, 0), dpi=400)
    plt.axis('off')
    n = 500
    for k in range(15):
        X = BB(n, settings.rng)
        plt.plot(
            np.linspace(0, 2 * pi, n),
            [sin(X[i]) * cos(i) + k * 1.6 for i in range(n)],
            color=settings.rng.choice(color1 + color2),
            lw=settings.rng.uniform(1.5, 4),
            alpha=1,
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
