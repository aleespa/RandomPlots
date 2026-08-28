import gc
from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    ran1 = settings.rng.beta(1.5, 1, 1000) * 40 * pi
    fig = plt.figure(figsize=(13, 13), facecolor='black', dpi=200)
    plt.axis('off')
    line1 = [(t) * sin(t) * cos(t) for t in np.linspace(0, 80 * pi, 10000)]
    line2 = [(t) * cos(t) * t**2 for t in np.linspace(0, 80 * pi, 10000)]
    plt.plot(line1[: 27 * 360], line2[: 27 * 360], color='white')
    for i in range(360):
        plt.plot(
            line1[27 * i : 27 * (i + 1) + 1],
            line2[27 * i : 27 * (i + 1) + 1],
            lw=3,
            alpha=0.9,
            color=plt.cm.plasma(settings.rng.uniform(0, 1)),
        )
        settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
