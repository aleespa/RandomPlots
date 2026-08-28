from math import cos, sin, pi

import matplotlib
import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = matplotlib.cm.get_cmap('summer')
    for s in np.linspace(0.5, 15, 210):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        X = np.array([t**s * cos(t) for t in np.linspace(0, 40 * pi, 4000)])
        Y = np.array([t**s * sin(t) for t in np.linspace(0, 40 * pi, 4000)])
        h = (40 * pi) ** s
        plt.plot(X / h, Y / h, lw=3, color=colors(s / 15))
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
