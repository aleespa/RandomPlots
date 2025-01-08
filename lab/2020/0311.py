from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for u in np.linspace(0, 2 * pi, 60):
    for z in np.linspace(0, 2 * pi, 60):
        plt.scatter(
            [6 * cos(u), 6 * cos(u) + cos(z)],
            [sin(u), sin(u) + sin(z)],
            color=plt.cm.Blues(z / (2 * pi)),
            s=35,
        )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/11032020.png', facecolor='black')
