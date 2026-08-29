from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='white', dpi=400)
    p = plt.axis('off')
    for u in np.linspace(0, 2 * pi, 40):
        for z in np.linspace(0, 2 * pi, 40):
            plt.plot(
                [cos(u), cos(u) + cos(z)],
                [sin(u), sin(u) + sin(z)],
                color=plt.cm.hsv(z / (2 * pi)),
                alpha=0.7,
                lw=1.3,
            )
    settings.save_frame('white')

if __name__ == '__main__':
    generate()
