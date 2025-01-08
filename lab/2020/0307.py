from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for z in np.linspace(0, 1, 50):
    plt.plot(
        [z * cos(x) * sin(x) for x in np.linspace(1, 3 * pi, 500)],
        [z * cos(x) * sin(x * 2) for x in np.linspace(1, 3 * pi, 500)],
        alpha=0.9,
        color=plt.cm.RdPu(z),
        lw=2,
    )

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/07032020.PNG', facecolor='black'
)
