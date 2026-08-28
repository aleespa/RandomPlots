from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    m = 0
    lws = settings.rng.normal(7, 1, 50)
    cols = settings.rng.uniform(0, 1, 50)
    for b in np.linspace(0, 2 * pi, 300):
        plt.figure(figsize=(13, 13), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(-2.1, 2.1)
        plt.ylim(-2.1, 2.1)
        i = 0
        for z in np.linspace(1, 7, 50):
            plt.plot(
                [
                    cos(t) * z * (1 - cos(t * 5) * sin(2 * t))
                    for t in np.linspace(0, 2 * pi, 500)
                ],
                [
                    sin(t) * z * (1 - cos(t * 5) * sin(b + 2 * t))
                    for t in np.linspace(0, 2 * pi, 500)
                ],
                lw=lws[i],
                alpha=0.75,
                color=plt.cm.Blues(cols[i]),
            )
            i += 1
        settings.save_frame('black')
        m += 1


if __name__ == '__main__':
    generate()
