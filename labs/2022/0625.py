import numpy as np
import matplotlib.pylab as plt
from math import sqrt, pi
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]

n = 30
sides = 12
r = np.linspace(0.65, 1, n)
points = []
for s in range(sides):
    points.append((r * np.cos(s * 2 * pi / sides), r * np.sin(s * 2 * pi / sides)))

plt.figure(num=1, clear=True, figsize=(14, 14), dpi=400, facecolor='black')
plt.axis('off')
for j in range(n):
    for s in range(sides):
        if s % 4 == 0:
            colors = plt.cm.autumn(j / n)
        elif s % 4 == 1:
            colors = plt.cm.summer(j / n)
        elif s % 4 == 2:
            colors = plt.cm.winter(j / n)
        else:
            colors = plt.cm.spring(j / n)
        plt.plot([points[s % sides][0][j], points[(s + 1) % sides][0][n - j - 1]],
                 [points[s % sides][1][j], points[(s + 1) % sides][1][n - j - 1]],
                 color=colors, lw=1)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(f"./../outputs/{filename}_v1.png", facecolor='black', )
plt.close()
