from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    m = 0
    for y in np.linspace(0, 18, 390):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(-2.1, 2.1)
        plt.ylim(-2.1, 2.1)
        plt.plot(
            [cos(t) * (1 - cos(t * y) * sin(y * t)) for t in np.linspace(0, 2 * pi, 500)],
            [sin(t) * (1 - cos(t * y) * sin(y * t)) for t in np.linspace(0, 2 * pi, 500)],
            lw=7,
            color=plt.cm.PiYG(y / 18),
        )
        settings.save_frame('black')
        m += 1


if __name__ == '__main__':
    generate()
