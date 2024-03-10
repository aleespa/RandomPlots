from __future__ import division

from math import sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np


def browninan_bridge(n):
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
n = 500
for k in range(35):
    X = browninan_bridge(n)
    p = plt.plot(np.linspace(0, 1, n),
                 [X[i] + k * i / n - k for i in range(n)],
                 color=np.random.choice(color1),
                 lw=2, alpha=0.8)
for k in range(35):
    X = browninan_bridge(n)
    p = plt.plot(np.linspace(0, 1, n),
                 [X[i] + (k - 34) * i / n - k for i in range(n)],
                 color=np.random.choice(color2),
                 lw=2, alpha=0.8)
p = plt.savefig('C:/Users/Alejandro/Pictures/RandomPlots/09112019.png',
                facecolor='black')
