from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def dibujo(t, n):
    plt.plot(
        [i * cos(t * i) for i in range(n)],
        [i * sin(t * i) for i in range(n)],
        color='white',
    )
    for i in range(n):
        plt.scatter(
            [i * cos(t * i)],
            [i * sin(t * i)],
            zorder=3,
            color=plt.cm.cool(1 - i / n),
            s=65,
        )

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = ['#32886f', '#0b5a42', '#72b7a3', '#bbbbbb', '#aaaaaa'] * 100
    n = 0




    n = 50
    m = 0
    for t in np.linspace(0, 2 * pi, 600):
        p = plt.figure(figsize=(12, 12), facecolor='black')
        p = plt.axis('off')
        p = plt.xlim(-n, n)
        p = plt.ylim(-n, n)
        dibujo(t, n)
        settings.save_frame('black')
        m += 1

if __name__ == '__main__':
    generate()
