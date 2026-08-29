from math import sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    for u, i in zip(np.linspace(0, 12, 450), range(450)):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')
        for z in np.linspace(0, 2, 50):
            plt.plot(
                np.linspace(z, 2 * pi - z, 300),
                [0.5 * sin(x + z * u) + z for x in np.linspace(0, 2 * pi, 300)],
                lw=2.5,
                color=plt.cm.spring(z / 2),
            )
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
