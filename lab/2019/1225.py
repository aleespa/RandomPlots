import gc
from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for _ in range(15):
        plt.plot(
            settings.rng.uniform(-1, 1, 100),
            settings.rng.uniform(-1, 1, 100),
            lw=1,
            color=plt.cm.rainbow(settings.rng.uniform(0, 1) - 0.1),
            alpha=0.75,
        )
    for z in np.linspace(0.9, 0, 15):
        plt.plot(
            [z * cos(t) for t in np.linspace(0, 2 * pi, 600)],
            [z * sin(t) for t in np.linspace(0, 2 * pi, 600)],
            lw=14,
            alpha=z,
            color='black',
        )
    settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
