from math import sqrt

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')


def brownian_path(N):
    Δt_sqrt = sqrt(1 / N)
    Z = np.random.randn(N)
    Z[0] = 0
    B = np.cumsum(Δt_sqrt * Z)
    return B


X = brownian_path(80)
for c in np.linspace(0, 0.5):
    plt.plot(c * X + c * abs(min(X)), color='white', lw=0.5, zorder=50)
for c in np.linspace(0, 0.5):
    plt.plot(-(c * X + c * abs(min(X))), color='red', lw=0.5)
plt.show()
# plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/24022020.png',facecolor='black')
