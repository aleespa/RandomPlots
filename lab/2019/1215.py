from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = [
        '#9be3f2',
        '#4d9447',
        '#16a3f9',
        '#66a33d',
        '#73ab38',
        '#87ae5f',
        '#AE5F5F',
    ]
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for y in range(4, 130):
        plt.scatter(
            [cos(x) * y * cos((2 * pi * y) / 129) for x in np.linspace(0, 2 * pi, y)],
            [sin(x) * y for x in np.linspace(0, 2 * pi, y)],
            s=10,
            alpha=0.7,
            color=settings.rng.choice(colors),
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
