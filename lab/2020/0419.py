from math import sqrt

import matplotlib.pylab as plt
import numpy as np


def brownian_path(N):
    Δt_sqrt = sqrt(1 / N)
    Z = np.random.randn(N)
    Z[0] = 0
    B = np.cumsum(Δt_sqrt * Z)
    return B


p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
# plt.ylim(-6,6)
colors = ['#0099ff', '#09bfe8', '#17fff9', '#09e8ad', '#0aff86']
for u in np.linspace(0, 1, 30):
    for i, k in enumerate([10, 50, 100, 500, 1000]):
        X = brownian_path(k) + i * 4
        c = np.random.choice(colors)
        plt.plot(np.linspace(0, 1, k), X, lw=2 / (i + 1), color=c, alpha=0.8)
        plt.plot(np.linspace(1, 2, k), X[::-1], lw=2 / (i + 1), color=c, alpha=0.8)

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/19042020.PNG', facecolor='black'
)
