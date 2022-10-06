import numpy as np
import matplotlib.pylab as plt
from numpy import sqrt, pi, cos, sin
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]


def ikeda_map(x0, y0, u):
    tn = 0.4 - 6 / (1 + x0 ** 2 + y0 ** 2)
    x1 = 1 + u * (x0 * cos(tn) - y0 * sin(tn))
    y1 = u * (x0 * sin(tn) + y0 * cos(tn))
    return x1, y1


u = 0.918
m = 1000
n = 200
Z_initial = np.random.normal(0, 3, size=(2, m))
Z = np.zeros((2, m, n))
Z[:,:,0] = Z_initial
fig = plt.figure(figsize=(12, 12), dpi=50)
ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
for i in range(n - 1):
    Z[0, :, i + 1], Z[1, :, i + 1] = ikeda_map(Z[0, :, i], Z[1, :, i], u)
for j in range(m):
    plt.plot(Z[0, j, :], Z[1, j, :], color='w', lw=1, alpha=0.2)
plt.show()
