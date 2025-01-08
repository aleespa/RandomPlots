from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for t in np.linspace(0, 2 * pi, 90):
    plt.plot(
        [cos(x) * x for x in np.linspace(0, 2 * t * pi, 1000)],
        [x * sin(x + t) for x in np.linspace(0, 2 * t * pi, 1000)],
        alpha=1,
        color=plt.cm.RdPu(t / (2 * pi)),
        zorder=1500 - int(t * 100),
        lw=2,
    )

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/13052020.PNG', facecolor='black'
)
