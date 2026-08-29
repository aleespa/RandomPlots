from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    for i, a in enumerate(np.linspace(0, 0.07, 390)):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')
        N = 450
        p = plt.ylim(-21, 21)
        p = plt.xlim(-21, 21)
        plt.scatter(
            [(sqrt(t) - 0.5) * cos(2 * pi * t * a) for t in range(N)],
            [(sqrt(t) - 0.5) * sin(2 * pi * t * a) for t in range(N)],
            s=110,
            color=[plt.cm.viridis(1 - a * u * (1 / 0.2)) for u in np.linspace(0, 1, N)],
        )
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
