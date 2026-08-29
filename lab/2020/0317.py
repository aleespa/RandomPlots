from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
    p = plt.axis('off')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    for t in np.linspace(0, pi):
        plt.scatter(
            [x * cos(x + t) for x in np.linspace(0, 50, 1000)],
            [x * sin(x) for x in np.linspace(0, 51, 1000)],
            s=4,
            color=plt.cm.YlGnBu(t / pi),
            alpha=0.7,
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
