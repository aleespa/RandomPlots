from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='white', dpi=400)
p = plt.axis('off')
for u in np.linspace(0, 2 * pi, 40):
    for z in np.linspace(0, 2 * pi, 40):
        plt.plot(
            [cos(u), cos(u) + cos(z)],
            [sin(u), sin(u) + sin(z)],
            color=plt.cm.hsv(z / (2 * pi)),
            alpha=0.7,
            lw=1.3,
        )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10032020.png', facecolor='white')
