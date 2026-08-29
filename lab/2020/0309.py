from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for z in np.linspace(0, 1, 50):
        plt.scatter(
            [z * cos(x) * sin(x) * sin(3 * x) for x in np.linspace(1, 3 * pi, 500)],
            [z * cos(x) * sin(x) * cos(3 * x) for x in np.linspace(1, 3 * pi, 500)],
            alpha=0.9,
            color=plt.cm.gist_heat_r(z - 0.1),
            s=z * 8,
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
