from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def brownian_path(N):
    Δt_sqrt = sqrt(1 / N)
    Z = np.random.randn(N)
    Z[0] = 0
    B = np.cumsum(Δt_sqrt * Z)
    return B

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')




    n = 800
    X, Y = brownian_path(n), brownian_path(n)
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    for i in range(n - 1):
        u = 0.5 * sqrt((X[i] - X[i + 1]) ** 2 + (Y[i] - Y[i + 1]) ** 2)
        plt.plot(
            [z * cos(z) + 10 * X[i] for z in np.linspace(0, 4 * pi, 120)],
            [z * sin(z) + Y[i] for z in np.linspace(0, 4 * pi, 120)],
            alpha=0.7,
            lw=settings.rng.uniform(0.5, 4),
            color=plt.cm.magma(settings.rng.uniform(0, 1)),
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
