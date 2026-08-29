from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for l in np.linspace(0, 2 * pi, 100):
        p = plt.plot(
            [
                sqrt((cos(l) - 1) ** 2 + (sin(l) - 1) ** 2) * cos(z) + cos(l) - 1
                for z in np.linspace(0, 2 * pi)
            ],
            [
                sqrt((cos(l) - 1) ** 2 + (sin(l) - 1) ** 2) * sin(z) + sin(l) - 1
                for z in np.linspace(0, 2 * pi)
            ],
            lw=settings.rng.uniform(0.5, 5),
            alpha=0.7,
            color=plt.cm.RdPu(settings.rng.uniform(0, 1)),
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
