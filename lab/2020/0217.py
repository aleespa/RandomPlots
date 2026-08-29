from math import sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def BB(n):
    T = np.linspace(0, 1, n)
    B = np.ones(n) * 0
    for i in range(n):
        xi = sqrt(2) * np.random.randn() / ((i + 1) * pi)
        B = B + xi * np.array([sin((i + 1) * pi * t) for t in T])
    return B

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)





    n = 1000
    color1 = ['#96ceb4', '#ffeead', '#ff6f69', '#ffcc5c', '#88d8b0']
    color2 = ['#f7f4a3', '#7fccec', '#6a81d9', '#a479c9', '#dfdfdf']

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for k in range(55):
        X = BB(n)
        p = plt.plot(
            np.linspace(0, 1, n),
            [X[i] + (k) * i / n - k for i in range(n)],
            color=settings.rng.choice(color1),
            lw=1,
            alpha=0.8,
        )
        p = plt.plot(
            np.linspace(0, 1, n),
            [-(X[i] + (k) * i / n - k) for i in range(n)],
            color=settings.rng.choice(color1),
            lw=1,
            alpha=0.8,
        )
        p = plt.plot(
            -np.linspace(0, 1, n),
            [-(X[i] * 6 + (k) * i / n - k) for i in range(n)],
            color=settings.rng.choice(color2),
            lw=0.5,
            alpha=1,
        )
        p = plt.plot(
            -np.linspace(0, 1, n),
            [(X[i] * 6 + (k) * i / n - k) for i in range(n)],
            color=settings.rng.choice(color2),
            lw=0.5,
            alpha=1,
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
