from math import exp

import matplotlib.pylab as plt
import numpy as np

U1, U2 = np.random.uniform(0, 1, 200), np.random.uniform(0, 1, 200)
sizes = np.array([np.random.uniform(80, 550) * t for t in range(1, 201)])
norm = lambda x, t: exp(-((x - t) ** 2) / 550) * (x < t)
colors = (
    ['#ff6f4b'] * 20
    + ['#fd4c55'] * 20
    + ['#e13661'] * 20
    + ['#c1246b'] * 20
    + ['#a11477'] * 20
    + ['#c1246b'] * 20
    + ['#e13661'] * 20
    + ['#fd4c55'] * 20
    + ['#ff6f4b'] * 40
)

for t in range(200):
    p = plt.figure(figsize=(12, 12), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(0, 1)
    p = plt.ylim(0, 1)
    p = plt.scatter(
        U1, U2, s=[2600 * norm(x, t) for x in range(200)], alpha=0.7, color=colors
    )
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/14112019/fig{t}.png',
        facecolor='black',
    )
