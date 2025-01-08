from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
p = plt.axis('off')
for z in np.linspace(0, 2 * pi, 20):
    plt.plot(
        [cos(x + z) * x for x in np.linspace(0, 22 * pi, 18)],
        [sin(x + z) * x for x in np.linspace(0, 22 * pi, 18)],
        lw=1.2,
        color=plt.cm.hot(1 - z / (2 * pi)),
        zorder=np.random.randint(2),
    )
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/22032020', facecolor='black')
