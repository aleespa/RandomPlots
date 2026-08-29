from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
    p = plt.axis('off')
    plt.xlim(-290, 290)
    plt.ylim(-290, 290)
    colors = ['#0ac7f0', '#00cc89', '#fdea2d', 'white', '#b7c3c1']
    for z in np.linspace(0, 250, 35):
        h = settings.rng.choice([-1, 0, 1], 10)
        f = np.random.binomial(10, 0.5, 10)
        X = np.linspace(0, 4 * pi, 500)
        plt.plot(
            [x * cos(x) * (sum([h[i] * cos(f[i] * x) for i in range(10)]) + 15) for x in X],
            [x * sin(x) * (sum([h[i] * cos(f[i] * x) for i in range(10)]) + 15) for x in X],
            color=settings.rng.choice(colors),
        )

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
