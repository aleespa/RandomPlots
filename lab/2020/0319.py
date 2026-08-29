from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    n = 80
    p = plt.xlim(-5, 5)
    p = plt.ylim(-5, 5)
    for i in range(n):
        r = settings.rng.uniform(2, 5)
        c = settings.rng.uniform(0, 1)
        plt.plot(
            [
                cos(x) * r
                for x in np.linspace(i * (2 * pi) / n, (i + 1) * (2 * pi) / n, 200)
            ],
            [
                sin(x) * r
                for x in np.linspace(i * (2 * pi) / n, (i + 1) * (2 * pi) / n, 200)
            ],
            color=plt.cm.autumn(1 - c),
            lw=2,
        )
        plt.plot(
            [cos(i * (2 * pi) / n) * r, 0],
            [sin(i * (2 * pi) / n) * r, 0],
            color=plt.cm.winter(c),
            lw=2,
        )
        plt.plot(
            [cos((i + 1) * (2 * pi) / n) * r, 0],
            [sin((i + 1) * (2 * pi) / n) * r, 0],
            color=plt.cm.winter(c),
            lw=2,
        )


    settings.save_frame('black')

if __name__ == '__main__':
    generate()
