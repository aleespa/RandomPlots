from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def cubo(a, b, t, color):
    p = plt.plot(
        [
            a * cos(t) - a * sin(t),
            b * cos(t) - a * sin(t),
            b * cos(t) - b * sin(t),
            a * cos(t) - b * sin(t),
            a * cos(t) - a * sin(t),
        ],
        [
            a * sin(t) + a * cos(t),
            b * sin(t) + a * cos(t),
            b * sin(t) + b * cos(t),
            a * sin(t) + b * cos(t),
            a * sin(t) + a * cos(t),
        ],
        color=plt.cm.twilight_shifted(a / (35 * pi) - 0.1),
        alpha=1,
        lw=3,
    )

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    p = plt.xlim(-300, 300)
    p = plt.ylim(-300, 300)




    for t in np.linspace(0, 35 * pi, 150):
        cubo(t, t * 2, 0.1 * t, 'red')

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
