from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
p = plt.axis('off')
p = plt.ylim(-2, 2)
p = plt.plot(
    np.linspace(0, 2 * pi), [sin(x) for x in np.linspace(0, 2 * pi)], color='red', lw=4
)
for i, z in enumerate(np.linspace(0, 2 * pi, 200)):
    p = plt.plot(
        np.linspace(0, 2 * pi),
        [sin(z) + cos(z) * (x - z) for x in np.linspace(0, 2 * pi)],
        color=plt.cm.summer(z / (2 * pi)),
    )
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/10042020/plot{i}.PNG',
        facecolor='black',
        lw=3,
    )
