from math import pi, tanh

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for z in np.linspace(-8, 8, 500):
        plt.plot(
            [tanh(x * z) for x in np.linspace(-pi, pi, 150)],
            lw=settings.rng.uniform(0.5, 4),
            alpha=0.75,
            color=plt.cm.winter(settings.rng.uniform(0.2, 1)),
        )
    plt.plot([tanh(x) for x in np.linspace(-pi, pi, 150)], lw=5, alpha=0.8, color='red')
    plt.plot([tanh(-x) for x in np.linspace(-pi, pi, 150)], lw=5, alpha=0.8, color='red')

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
