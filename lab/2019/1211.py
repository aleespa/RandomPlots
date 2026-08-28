from math import sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 6000
    m = 50
    fig = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(-sqrt(n), sqrt(n))
    plt.ylim(-sqrt(n), sqrt(n))
    for j in range(m):
        X = np.array([0, 0])
        S = [X]
        for i in range(n):
            X = X + settings.rng.normal(0, 0.5, 2)
            S.append(X)
        plt.plot(
            [S[i][0] for i in range(n)],
            [S[i][1] for i in range(n)],
            alpha=0.6,
            color=plt.cm.hsv(j / m),
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
