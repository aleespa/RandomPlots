from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = [plt.cm.YlGn(u) for u in settings.rng.uniform(0, 1, 50)]
    for j, z in enumerate(np.linspace(0, 2 * pi, 210)):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')

        for i, t in enumerate(np.linspace(0, 2 * pi, 50)):
            plt.plot(
                [0, cos(t), cos(z)],
                [0, sin(t), sin(z)],
                color=colors[i],
                lw=3.5,
                alpha=0.75,
            )

        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
