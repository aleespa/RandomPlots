from math import cos, sin

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    for l, i in zip(np.linspace(0, 3, 690), range(690)):
        p = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
        p = plt.axis('off')
        p = plt.xlim(-40, 40)
        p = plt.ylim(-10, 470)
        for t in np.linspace(0, 0.5, 40):
            plt.plot(
                [x * cos(x * l) for x in np.linspace(0, 40, 400)],
                [x * sin(x * t) + 10 * x for x in np.linspace(0, 40, 400)],
                alpha=0.6,
                lw=settings.rng.uniform(0.5, 3),
                color=plt.cm.gray(t * 2),
            )
        settings.save_frame('black')

if __name__ == '__main__':
    generate()
