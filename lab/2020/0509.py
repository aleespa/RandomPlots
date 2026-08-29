from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def circles(x, y, r, l):
    c = plt.cm.cool(np.random.uniform(0, 1))
    for R in np.linspace(0, 1, 20):
        plt.plot(
            [r * R * cos(t) + x for t in np.linspace(0, 2 * pi, 6)],
            [r * R * sin(t) + y for t in np.linspace(0, 2 * pi, 6)],
            zorder=l + 1,
            color=c,
            lw=2,
        )
        ax.add_artist(plt.Circle((x, y), radius=r, color="black", zorder=l))

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    fig, ax = plt.subplots(figsize=(14, 14), facecolor='black', dpi=400)

    p = plt.axis('off')
    p = plt.xlim(0, 2)
    p = plt.ylim(0, 2)




    for i in range(100):
        circles(settings.rng.uniform(-0.2, 2.2), settings.rng.uniform(0, 2), 0.5, i)

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
