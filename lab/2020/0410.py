from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    fig = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
    p = plt.axis('off')
    p = plt.ylim(-2, 2)
    p = plt.plot(
        np.linspace(0, 2 * pi), [sin(x) for x in np.linspace(0, 2 * pi)], color='red', lw=4
    )
    for i, z in enumerate(np.linspace(0, 2 * pi, 200)):
        p = plt.plot(
            np.linspace(0, 2 * pi),
            [sin(z) + cos(z) * (x - z) for x in np.linspace(0, 2 * pi)],
            color=plt.cm.summer(z / (2 * pi)),
        )
        p = settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
