from math import sin, pi

import matplotlib.pylab as plt
import numpy as np

for u, i in zip(np.linspace(0, 1, 420), range(420)):
    p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
    p = plt.axis('off')
    for z in np.linspace(0, 2, 20):
        plt.plot(
            np.linspace(z, 2 * pi - z, 300),
            [
                0.5 * sin(x * u * 5 + u * 2 * pi) + z
                for x in np.linspace(0, 2 * pi, 300)
            ],
            lw=8 / (z**1.7 + 1),
            zorder=20 - z,
            alpha=0.95,
            color=plt.cm.Greys(z / 2),
        )
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/27032020/plot{i}.PNG',
        facecolor='black',
    )
