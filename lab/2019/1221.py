import gc
from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = ['#ffdb00', '#f64b4b', '#ff7429', '#ff8e02', '#ffb910'] * 10
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    i = 0
    for z in np.linspace(0, 10, 15):
        plt.plot(
            [cos(t * 6) * z for t in np.linspace(0, 2 * pi, 600)],
            [sin(t * 10) * z for t in np.linspace(0, 2 * pi, 600)],
            lw=settings.rng.uniform(4, 10),
            alpha=0.8,
            color=colors[i],
        )
        i += 1
    settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
