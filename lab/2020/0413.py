from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    for i, u in enumerate(
        list(np.linspace(0, 0.9999, 120)) + list(np.linspace(0, 0.9999, 120))[::-1]
    ):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')
        plt.plot(
            [u**2 * x + cos(x) for x in np.linspace(0, 25 * pi, 2000)],
            [u**2 * x - sin(x) for x in np.linspace(0, 25 * pi, 2000)],
            color=plt.cm.Reds(u**2 - 0.05),
            lw=10 * u,
        )
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
