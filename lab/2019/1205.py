from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    for y in np.linspace(-pi, pi, 240):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.scatter(
            [x * cos(x - y) for x in range(2000)],
            [x * sin(x) for x in range(2000)],
            s=6,
            color=[plt.cm.autumn(1 - x) for x in np.linspace(0, 1, 2000)],
            alpha=0.7,
        )
        plt.scatter(
            [x * cos(x + y) for x in range(2000)],
            [x * sin(x) for x in range(2000)],
            s=6,
            color=[plt.cm.summer(1 - x) for x in np.linspace(0, 0.9, 2000)],
            alpha=0.7,
        )
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
