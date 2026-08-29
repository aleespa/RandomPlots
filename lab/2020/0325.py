from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = ['#23bbad', '#25d9c8', '#2abed9', '#ff6da2', '#f92672'] * 12
    for u, i in zip(np.linspace(0, 2 * pi, 360), range(360)):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')
        for z in np.linspace(0, 1):
            p = plt.scatter(
                [z * cos(x) * sin(u + x) for x in np.linspace(0, 2 * pi, 60)],
                [z * cos(x + z * pi / 2) for x in np.linspace(0, 2 * pi, 60)],
                color=colors,
                lw=0.5,
                s=55,
                alpha=0.8,
            )
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
