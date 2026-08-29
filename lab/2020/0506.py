from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    for i, a in enumerate(np.linspace(1, 2, 360)):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')
        for t in np.linspace(0, 20 * pi, 180):
            plt.plot([2 * cos(t), cos(a * t)], [2 * sin(t), sin(a * t)], color='white')
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
