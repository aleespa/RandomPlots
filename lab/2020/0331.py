from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    colors = ['#fe60a1', '#c961f7', '#ff734c', '#3bc7ff', '#8089ff']
    for u in np.linspace(0, 2 * pi, 20):
        p = plt.plot(
            [2 * cos(4 * x) * sin(x) for x in np.linspace(0, 2 * pi, 2000)],
            [2 * sin(x) * cos(x) * u for x in np.linspace(0, 2 * pi, 2000)],
            lw=2,
            alpha=0.8,
            zorder=settings.rng.choice([0, 1, 2]),
            color=settings.rng.choice(colors),
        )
    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
