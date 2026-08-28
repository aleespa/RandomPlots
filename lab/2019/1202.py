from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 0
    for y in np.linspace(0.1, 35, 210):
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.axis('off')
        plt.plot(
            [cos(x) for x in np.linspace(0, 2 * pi, 4000)],
            [cos(x**2) * sin(x) for x in np.linspace(0, y * pi, 4000)],
            lw=0.9,
            alpha=0.8,
            color=plt.cm.hsv(n / 420),
        )
        settings.save_frame('black')
        n += 1

    for y in np.linspace(0.1, 35, 210):
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.axis('off')
        plt.plot(
            [cos(x) for x in np.linspace(0, 2 * pi, 4000)],
            [cos(x**2) * sin(x) for x in np.linspace(0, (35 - y) * pi, 4000)],
            lw=0.9,
            alpha=0.8,
            color=plt.cm.hsv(n / 420),
        )
        settings.save_frame('black')
        n += 1


if __name__ == '__main__':
    generate()
