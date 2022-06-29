import numpy as np
import matplotlib.pylab as plt
from math import sqrt, pi
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]

n = 200
sides = 12
x = np.linspace(0, 2*pi, n)
points = []
points.append((x, np.sin(x)))
points.append((x, 1+np.cos(x)))

plt.figure(num=1, clear=True, figsize=(14, 14), dpi=400, facecolor='black')
plt.axis('off')
for j in range(n):
    for s in range(1):
        plt.plot([points[s % sides][0][j], points[(s + 1) % sides][0][n - j - 1]],
                 [points[s % sides][1][j], points[(s + 1) % sides][1][n - j - 1]], lw=1)

# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
plt.savefig(f"./../outputs/{filename}.png", facecolor='black', )
plt.close()
