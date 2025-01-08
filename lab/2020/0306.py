from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
plt.ylim(-1.2, 1.2)
plt.xlim(-1.2, 1)
for z in np.linspace(0.001, 1, 40):
    a = 1
    b = z
    c = sqrt(a**2 - b**2)
    plt.scatter(
        [(cos(x) - c * sin(x) ** 2) for x in np.linspace(0, 2 * pi, 500)],
        [
            ((a**2 - 2 * c**2 + a * c * cos(x)) * sin(x)) / b
            for x in np.linspace(0, 2 * pi, 500)
        ],
        s=9,
        alpha=0.7,
        color=plt.cm.GnBu(z),
    )
for z in np.linspace(0.7, 0.75, 15):
    a = 1
    b = z
    c = sqrt(a**2 - b**2)
    plt.scatter(
        [(cos(x) - c * sin(x) ** 2) for x in np.linspace(0, 2 * pi, 1000)],
        [
            ((a**2 - 2 * c**2 + a * c * cos(x)) * sin(x)) / b
            for x in np.linspace(0, 2 * pi, 1000)
        ],
        s=10,
        alpha=1,
        color='black',
    )
p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/06032020.PNG', facecolor='black'
)
