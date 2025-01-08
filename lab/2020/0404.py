from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
plt.plot(
    [cos(x) for x in np.linspace(0, 2 * pi)],
    [sin(x) for x in np.linspace(0, 2 * pi)],
    lw=2,
    color=plt.cm.BuPu(np.random.uniform(0, 1)),
)
for k in range(18):
    for t in np.linspace(0, 2 * pi, int(k * 2 * pi)):
        plt.plot(
            [cos(x) + cos(t) * 2 * k for x in np.linspace(0, 2 * pi)],
            [sin(x) + sin(t) * 2 * k for x in np.linspace(0, 2 * pi)],
            lw=2,
            color=plt.cm.BuPu(np.random.uniform(0, 1)),
        )
p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/04042020.PNG', facecolor='black'
)
