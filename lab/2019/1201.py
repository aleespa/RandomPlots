from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for x in np.linspace(-2 * pi, 2 * pi, 300):
        plt.plot(
            [-x, x],
            [cos(-x), sin(x + 5)],
            lw=0.9,
            color=plt.cm.Spectral((x + 2 * pi) / (4 * pi)),
            alpha=0.8,
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
