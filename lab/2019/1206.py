from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def cubo(a, b, t):
    plt.plot(
        [
            a * cos(t) - a * sin(t),
            b * cos(t) - a * sin(t),
            b * cos(t) - b * sin(t),
            a * cos(t) - b * sin(t),
            a * cos(t) - a * sin(t),
        ],
        [
            a * sin(t) + a * cos(t),
            b * sin(t) + a * cos(t),
            b * sin(t) + b * cos(t),
            a * sin(t) + b * cos(t),
            a * sin(t) + a * cos(t),
        ],
        color=plt.cm.hsv(a / (1)),
        alpha=0.9,
        lw=1,
    )


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
    plt.axis('off')

    for t in np.linspace(0, 2 * pi, 100):
        cubo(cos(t + 0.1) ** 2, sin(t + 0.5) ** 2, pi / 4)
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
