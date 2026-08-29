from math import sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    n = 2000
    X, Y = settings.rng.uniform(0, 1, n), settings.rng.uniform(0, 1, n)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for i in range(n):
        j = np.argmin(
            [
                sqrt((X[i] - X[m]) ** 2 + (Y[i] - Y[m]) ** 2) * (m != i) + 10 ** (m == i)
                for m in range(n)
            ]
        )
        plt.plot([X[i], X[j]], [Y[i], Y[j]], lw=2, color='#39a2ae')
    settings.save_frame('black')

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    i = 0
    lista = list(range(n))
    for l in range(n):
        j = np.argmin(
            [
                sqrt((X[i] - X[m]) ** 2 + (Y[i] - Y[m]) ** 2) * (m in lista)
                + 10 * (m not in lista)
                for m in range(n)
            ]
        )
        if sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2) < 0.05:
            plt.plot([X[i], X[j]], [Y[i], Y[j]], lw=2, color='#ff4500')
        lista.remove(j)
        i = j
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
