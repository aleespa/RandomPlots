from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for z in np.linspace(0, 5, 50):
        plt.plot(
            [cos(t + z) * z for t in np.linspace(0, 2 * pi, 4)],
            [sin(t + z) * z for t in np.linspace(0, 2 * pi, 4)],
            lw=3,
            color=plt.cm.Spectral(z / 5),
        )

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
