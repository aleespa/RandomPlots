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
        color=plt.cm.PuRd(a / (20 * pi) - 0.3),
        alpha=1,
        lw=1.8,
    )


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(14, 14), facecolor='black', dpi=600)
    plt.axis('off')
    plt.xlim(-280, 280)
    plt.ylim(-280, 280)

    for t in np.linspace(0, 20 * pi, 120):
        cubo(t, 3 * t + 1, t * 4.465114832535881)
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
