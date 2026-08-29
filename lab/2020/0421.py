from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for t in range(110):
        plt.plot(
            [cos(x - t) * t for x in np.linspace(0, 2 * pi, 5)],
            [sin(x) * t for x in np.linspace(0, 2 * pi, 5)],
            color=plt.cm.BuPu(t / 100),
            zorder=120 - t,
            alpha=0.85,
            lw=3,
        )

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
