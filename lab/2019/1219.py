from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

m = 0
for y in np.linspace(0, 18, 390):
    p = plt.figure(figsize=(12, 12), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-2.1, 2.1)
    p = plt.ylim(-2.1, 2.1)
    plt.plot(
        [cos(t) * (1 - cos(t * y) * sin(y * t)) for t in np.linspace(0, 2 * pi, 500)],
        [sin(t) * (1 - cos(t * y) * sin(y * t)) for t in np.linspace(0, 2 * pi, 500)],
        lw=7,
        color=plt.cm.PiYG(y / 18),
    )
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/19122019/plot{m}.png',
        facecolor='black',
    )
    m += 1
