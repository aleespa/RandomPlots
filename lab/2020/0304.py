from math import sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    N = 14000
    X, Y = settings.rng.uniform(-1, 1, N), settings.rng.uniform(-1, 1, N)
    color = [sqrt(X[i] ** 2 + Y[i] ** 2) * 0.9 for i in range(N)]
    plt.scatter(X, Y, color=plt.cm.binary(color), s=8)

    for z in np.linspace(0.2, 1):
        b = [
            sqrt(X[i] ** 2 + Y[i] ** 2) < 0.8 * z and sqrt(X[i] ** 2 + Y[i] ** 2) > 0.74 * z
            for i in range(N)
        ]
        plt.scatter(X[b], Y[b], s=10, color=plt.cm.binary(z))

    b = [
        sqrt(X[i] ** 2 + Y[i] ** 2) < 0.8 and sqrt(X[i] ** 2 + Y[i] ** 2) > 0.75
        for i in range(N)
    ]
    plt.scatter(X[b], Y[b], s=10, color=plt.cm.spring(settings.rng.uniform(0, 1, sum(b))))
    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
