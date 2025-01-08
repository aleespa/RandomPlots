from math import sqrt

import matplotlib.pylab as plt
import numpy as np

colors = ['#ff9b9b', '#f8ff90', '#a9ff8f', '#22ba5a', '#58c0e7']
p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-1, 1)
p = plt.ylim(-1, 1)
for x in np.linspace(-1, 1, 121):
    p = plt.plot(
        [x, x * x], [0, sqrt(1 - x**2)], alpha=0.8, color=np.random.choice(colors)
    )
    p = plt.plot(
        [x, x * x], [0, -sqrt(1 - x**2)], alpha=0.8, color=np.random.choice(colors)
    )
    p = plt.plot(
        [x, x * x], [0, -sqrt(1 - x**2) * x], alpha=0.8, color=np.random.choice(colors)
    )
    p = plt.plot(
        [x, x * x], [0, sqrt(1 - x**2) * x], alpha=0.8, color=np.random.choice(colors)
    )
p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/15112019.png', facecolor='black'
)
