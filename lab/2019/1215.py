from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

colors = ['#9be3f2', '#4d9447', '#16a3f9', '#66a33d', '#73ab38', '#87ae5f', '#AE5F5F']
p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
for y in range(4, 130):
    plt.scatter(
        [cos(x) * y * cos((2 * pi * y) / 129) for x in np.linspace(0, 2 * pi, y)],
        [sin(x) * y for x in np.linspace(0, 2 * pi, y)],
        s=10,
        alpha=0.7,
        color=np.random.choice(colors),
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/15122019.png', facecolor='black')
