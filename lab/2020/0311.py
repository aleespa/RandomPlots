from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for u in np.linspace(0, 2 * pi, 60):
        for z in np.linspace(0, 2 * pi, 60):
            plt.scatter(
                [6 * cos(u), 6 * cos(u) + cos(z)],
                [sin(u), sin(u) + sin(z)],
                color=plt.cm.Blues(z / (2 * pi)),
                s=35,
            )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
