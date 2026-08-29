from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
    p = plt.axis('off')
    for z in np.linspace(0, 2 * pi, 20):
        plt.plot(
            [cos(x + z) * x for x in np.linspace(0, 22 * pi, 18)],
            [sin(x + z) * x for x in np.linspace(0, 22 * pi, 18)],
            lw=1.2,
            color=plt.cm.hot(1 - z / (2 * pi)),
            zorder=settings.rng.randomint(2),
        )
    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
