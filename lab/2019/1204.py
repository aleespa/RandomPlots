from math import cos, sin

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    X, Y = [x * cos(x) for x in range(30000)], [
        x * sin(x**1.0001) for x in range(30000)
    ]
    t = np.linspace(3000, 30000, 300)
    for i in range(50, 300):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
        plt.axis('off')
        plt.xlim(-t[i] - 500, t[i] + 500)
        plt.ylim(-t[i] - 500, t[i] + 500)
        plt.scatter(
            X,
            Y,
            s=[x**4 for x in np.linspace(0.5, 1.4, 30000)],
            color=[plt.cm.autumn(x) for x in np.linspace(0, 1, 30000)],
        )
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
