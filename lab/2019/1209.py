from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = ['#aaaaaa', '#e37398', '#f0c1cc', '#fee2e9', '#fe7ea8']

    fig = plt.figure(figsize=(14, 14), facecolor='black')
    plt.axis('off')
    for y in np.linspace(0, 2 * pi, 18):
        plt.plot(
            [cos(x) + cos(y) * 3 for x in np.linspace(0, 2 * pi, int(y * 2 * pi))],
            [sin(x) + sin(y) * 3 for x in np.linspace(0, 2 * pi, int(y * 2 * pi))],
            lw=5,
            alpha=0.9,
            color=settings.rng.choice(colors),
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
