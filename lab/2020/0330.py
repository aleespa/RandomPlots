from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    colors = ['#3ef406', '#ffd700', '#0552f4', '#4f9cf4', '#04a918']
    for u in np.linspace(0, 2 * pi, 20):
        p = plt.plot(
            [2 * sin(10 * x) * cos(x) for x in np.linspace(0, 2 * pi, 2000)],
            [2 * sin(10 * x) * sin(x) * u for x in np.linspace(0, 2 * pi, 2000)],
            lw=2,
            alpha=0.8,
            zorder=settings.rng.choice([0, 1, 2]),
            color=settings.rng.choice(colors),
        )
    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
