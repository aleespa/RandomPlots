from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    for r in [1, 0.5, 0.25]:
        for t in np.linspace(0, pi, 70):
            plt.plot(
                [
                    r * cos(0 + t),
                    r * cos((2 / 3) * pi + t),
                    r * cos(4 * pi / 3 + t),
                    r * cos((2) * pi + t),
                ],
                [
                    r * sin(0 + t),
                    r * sin((2 / 3) * pi + t),
                    r * sin(4 * pi / 3 + t),
                    r * sin((2) * pi + t),
                ],
                color=plt.cm.hsv(((t) / (pi))),
                alpha=0.8,
                lw=(r + 0.1),
            )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
