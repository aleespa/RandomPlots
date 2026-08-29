from math import cos, sin

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    m = 0
    for z in np.linspace(0, 10, 360):
        p = plt.figure(figsize=(13, 13), facecolor='black')
        p = plt.axis('off')
        colors = ['#ff1496', '#c3a6a8', '#ff2387', '#ff3278', '#fbabb0'] * 400
        plt.scatter(
            [i * cos(i) for i in range(2000)],
            [sin(i) * i**z for i in range(2000)],
            s=30,
            color=colors,
        )
        settings.save_frame('black')
        m += 1

if __name__ == '__main__':
    generate()
