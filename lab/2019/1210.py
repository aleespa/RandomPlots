from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    for y in np.linspace(0, 1, 600):
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.axis('off')
        plt.plot(
            [cos(x) for x in np.linspace(0, 25 * pi, 1000)],
            [sin(x) for x in np.linspace(0, 25 * (1 + y) * pi, 1000)],
            lw=4,
            alpha=1,
            color=plt.cm.autumn(y),
        )
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
