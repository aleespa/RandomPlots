from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
for _ in range(15):
    plt.plot(
        np.random.uniform(-1, 1, 100),
        np.random.uniform(-1, 1, 100),
        lw=1,
        color=plt.cm.rainbow(np.random.uniform(0, 1) - 0.1),
        alpha=0.75,
    )
for z in np.linspace(0.9, 0, 15):
    plt.plot(
        [z * cos(t) for t in np.linspace(0, 2 * pi, 600)],
        [z * sin(t) for t in np.linspace(0, 2 * pi, 600)],
        lw=14,
        alpha=z,
        color='black',
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/25122019.png', facecolor='black')
