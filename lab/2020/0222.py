from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

n = 40
colors = ['#ff6f69', '#ffcc5c', '#ffeead', '#ffffff', '#96ceb4'] * n
for j, k in zip(np.linspace(0, pi, 420), range(420)):
    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
    p = plt.axis('off')
    for i in range(n):
        plt.scatter(
            range(n),
            np.repeat(i, n),
            s=[100 * abs(cos((z / n) * 2 * pi * i * j)) for z in range(n)],
            alpha=0.8,
            color=colors[i],
        )
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/22022020/plot{k}.png',
        facecolor='black',
    )
