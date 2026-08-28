from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    for y in np.linspace(0, 2 * pi, 28):
        plt.scatter(
            [x * cos(x) + 200 * cos(y) for x in np.linspace(0, 200, 500)],
            [x * sin(x) + 200 * sin(y) for x in np.linspace(0, 200, 500)],
            s=[(x) for x in np.linspace(7, 4, 500)],
            color=[plt.cm.cool(x) for x in np.linspace(0, 0.5, 500)],
            alpha=1,
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
