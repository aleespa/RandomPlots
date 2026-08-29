from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = ['#32886f', '#0b5a42', '#72b7a3', '#bbbbbb', '#aaaaaa'] * 100
    n = 0
    m = 0
    for pol in [4, 5, 7]:
        p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
        p = plt.axis('off')
        for z in np.linspace(0, 2 * pi, 85):
            plt.plot(
                [cos(x) + 2 * cos(z) for x in np.linspace(0, 2 * pi, pol)],
                [sin(x) + 2 * sin(z) for x in np.linspace(0, 2 * pi, pol)],
                lw=6,
                alpha=0.75,
                color=colors[n],
            )
            n += 1
        settings.save_frame('black')
        m += 1

if __name__ == '__main__':
    generate()
