from math import sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582']
    CC = settings.rng.choice(colors, 60)
    for j in np.linspace(0, 12 * pi, 210):
        n = 0
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(0, 10)
        plt.ylim(-10, 10)
        for i in np.linspace(-12, 12, 60):
            plt.plot(
                np.linspace(0, 10),
                [sin(x + j) + i for x in np.linspace(0, 10)],
                color=CC[n],
                lw=5,
            )
            n += 1
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
