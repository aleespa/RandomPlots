from math import cos, sin, pi

import matplotlib
import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = matplotlib.cm.get_cmap('viridis')

    fig = plt.figure(figsize=(15, 15), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    for i in np.linspace(-pi, pi, 80):
        X = np.array([cos(t) + sin(i) for t in np.linspace(0, 2 * pi, 400)])
        Y = np.array([sin(t) + cos(i) for t in np.linspace(0, 2 * pi, 400)])
        plt.plot(
            X,
            Y,
            lw=settings.rng.choice([1, 1, 0.5, 0.7, 3, 4, 2]),
            alpha=settings.rng.uniform(0.5, 1),
            color=colors(settings.rng.uniform(0, 1)),
        )

    X = np.array([0.45 * cos(t) for t in np.linspace(0, 2 * pi, 400)])
    Y = np.array([0.45 * sin(t) for t in np.linspace(0, 2 * pi, 400)])
    plt.plot(X, Y, lw=25, alpha=0.8, color='black', zorder=100)
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
