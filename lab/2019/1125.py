import gc
from math import cos, sin, log, pi

import matplotlib
import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = matplotlib.cm.get_cmap('rainbow')
    X, Y = [log(t) * cos(t * 1.1) for t in np.linspace(10, 60 * pi, 4200)], [
        log(t) * sin(t) for t in np.linspace(10, 50 * pi, 4200)
    ]

    for i in range(208):
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.axis('off')
        plt.xlim(-5.3, 5.3)
        plt.ylim(-5.3, 5.3)
        plt.plot(
            X[20 * i : 20 * (i + 2)],
            Y[20 * i : 20 * (i + 2)],
            alpha=1,
            lw=3,
            color=colors(i / 210),
        )
        plt.scatter([X[20 * (i + 2)]], [Y[20 * (i + 2)]], color='white')
        settings.save_frame('black')

    fig = plt.figure(figsize=(12, 12), facecolor='black')
    plt.axis('off')
    plt.xlim(-5.3, 5.3)
    plt.ylim(-5.3, 5.3)

    for i in range(208):
        plt.plot(
            X[20 * i : 20 * (i + 2)],
            Y[20 * i : 20 * (i + 2)],
            alpha=0.9,
            lw=2.5,
            color=colors(i / 210),
        )
        settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
