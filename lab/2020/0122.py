from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def loc(z):
    p = plt.figure(figsize=(14, 14), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-1.1, 1.1)
    p = plt.ylim(-1.1, 1.1)
    for i, color in zip(np.linspace(0, 1, 60), colors):
        X, Y = [i * cos(y) * cos(z * y) for y in np.linspace(0, 2 * pi, 1000)], [
            i * cos(y) * sin(z * y) for y in np.linspace(0, 2 * pi, 1000)
        ]
        plt.plot(X, Y, alpha=0.85, lw=2.2, color=color)

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = [plt.cm.Spectral(settings.rng.uniform(0, 1)) for j in range(60)]




    for z, i in zip(np.linspace(0, 8, 300), range(300)):
        loc(z)
        settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
