from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    fig, ax = plt.subplots(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for t in np.linspace(0, 2 * pi, 90):
        plt.plot(
            [cos(x) * x for x in np.linspace(0, 2 * t * pi, 1000)],
            [x * sin(x + t) for x in np.linspace(0, 2 * t * pi, 1000)],
            alpha=1,
            color=plt.cm.RdPu(t / (2 * pi)),
            zorder=1500 - int(t * 100),
            lw=2,
        )

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
