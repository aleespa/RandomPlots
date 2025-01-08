import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt

n = 180
X, Y = np.random.uniform(-10, 10, n), np.random.uniform(-10, 10, n)


j = 0
for y in np.linspace(1, 0, 250):
    p = plt.figure(figsize=(12, 12), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-10.1, 10.1)
    p = plt.ylim(-10.1, 10.1)
    plt.scatter(X, Y, zorder=n**2, alpha=1, s=9, color='#ff7142')
    for i in range(n):
        p = plt.plot([X[i], X[i] * y], [Y[i], Y[i]], alpha=0.8, color='white', lw=2)
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/13122019/plot{j}.png',
        facecolor='black',
    )
    j += 1
for y in np.linspace(1, 0, 250):
    p = plt.figure(figsize=(12, 12), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-10.1, 10.1)
    p = plt.ylim(-10.1, 10.1)
    plt.scatter(X, Y, zorder=n**2, alpha=1, s=9, color='#ff7142')
    for i in range(n):
        plt.plot([X[i], 0], [Y[i], Y[i] * y], alpha=0.8, color='white', lw=2)
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/13122019/plot{j}.png',
        facecolor='black',
    )
    j += 1
