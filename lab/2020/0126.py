from math import sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np


def BB(n):
    T = np.linspace(0, 1, n)
    B = np.ones(n) * 0
    for i in range(n):
        xi = sqrt(2) * np.random.randn() / ((i + 1) * pi)
        B = B + xi * np.array([sin((i + 1) * pi * t) for t in T])
    return B


color1 = ['#96ceb4', '#ffeead', '#ff6f69', '#ffcc5c', '#88d8b0']
color2 = ['#f7f4a3', '#7fccec', '#6a81d9', '#a479c9', '#dfdfdf']
p = plt.figure(figsize=(14, 14), facecolor=(0, 0, 0), dpi=400)
p = plt.axis('off')
n = 1000
for k in range(40):
    X = BB(n)
    p = plt.plot(
        np.linspace(0, 1, n),
        [X[i] * (k / 8) - k for i in range(n)],
        color=plt.cm.GnBu(np.random.uniform(0, 1)),
        lw=2.2,
        alpha=0.8,
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/26012020.png', facecolor='black')
