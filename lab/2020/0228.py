from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for z in np.linspace(0, 1):
        plt.scatter(
            [z**2 * x * cos(x * z) for x in np.linspace(0, 20 * pi, 300)],
            -np.linspace(0, 10 * pi, 300),
            s=2.5,
            color=plt.cm.cool(z**2),
            alpha=0.8,
        )
        plt.scatter(
            [z**2 * x * cos(x * z) for x in np.linspace(0, 20 * pi, 300)],
            np.linspace(0, 10 * pi, 300) - 63,
            s=2.5,
            color=plt.cm.cool(1 - z**2),
            alpha=0.8,
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
