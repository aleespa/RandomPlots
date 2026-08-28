import gc
from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    X, Y = [t * cos(t) for t in np.linspace(0, 100 * pi, 10000)], [
        sin(t) for t in np.linspace(0, 100 * pi, 10000)
    ]
    colors = ['#eebdff', '#d59dee', '#a154c3', '#632b7c', '#5e0ba5']

    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(-pi * 100, pi * 100)
    plt.ylim(-1, 1)
    for i in range(180):
        plt.plot(
            X[i * 55 : (i + 1) * 55],
            Y[i * 55 : (i + 1) * 55],
            alpha=0.8,
            color=colors[i % 5],
        )
        settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
