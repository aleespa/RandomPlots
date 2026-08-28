from math import sin, pi

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from common.image_processing import ImageProcessingSettings


def BSp(x, y, n, Nij):
    z = 0
    for i in range(n):
        for j in range(n):
            z += (
                Nij[i][j]
                * (2 / (pi**2 * (i - 0.5) * (j - 0.5)))
                * sin((i - 0.5) * pi * x)
                * sin((j - 0.5) * pi * y)
            )
    return z


def BS(dt, rng):
    T = np.arange(0, 1 + dt, dt)
    Z = pd.DataFrame(columns=T[1:])
    n = len(T)
    Nij = rng.random((n, n))
    for x in T[1:]:
        Z[x] = [BSp(x, y, n, Nij) for y in T[1:]]
    return Z


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    z = BS(0.01, settings.rng).values

    for t in range(360):
        fig = plt.figure(figsize=(12, 12), dpi=200)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        ax.set_zlim(0.6, 2.1)
        plt.axis('off')
        ax.view_init(0, t)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y, sparse=True)
        ax.plot_surface(
            xx,
            yy,
            z,
            linewidth=0,
            antialiased=True,
            shade=True,
            cmap=cm.inferno,
            alpha=0.5,
        )
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
