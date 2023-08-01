import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy import sqrt, pi, cos, sin
from scipy import interpolate
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]

# R = 0.2*cos(np.linspace(0, 2 * pi, 6)) + 0.2*sin(np.linspace(0, 2 * pi, 6)) * 1j
R = np.array([1j, -1j, -1, 1])


def solver_polinomio(x):
    denominator = 1 / (x[..., np.newaxis] - R)
    return x - 1 / denominator.sum(axis=-1)


P = np.vectorize(solver_polinomio)

n = 3500
X = np.linspace(-1, 1, n)
Y = np.linspace(-1, 1, n)
xx, yy = np.meshgrid(X, Y)

Z0 = xx + yy * 1j
Z1 = Z0.copy()
distances = np.linalg.norm(Z1[..., np.newaxis] - R)
for _ in range(25):
    Z1 = np.where(distances.min(axis=-1) < 1e-12, Z1, solver_polinomio(Z1))
    distances = np.abs(Z1[..., np.newaxis] - R)
closest_idx = np.argmin(distances, axis=-1)

for i, colors in enumerate([[plt.cm.Greys(0.999), plt.cm.Greys(0.5)],
                            [plt.cm.Greys(0.2), plt.cm.Greys(0.2)],
                            [plt.cm.Greys(0.5), plt.cm.Greys(0.999)]]):
    cmap = ListedColormap(['k', 'k'] + colors)
    fig, _ = plt.subplots(figsize=(12, 12), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
    ax.imshow(closest_idx.reshape(n, n), extent=(-2, 2, -2, 2), cmap=cmap)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    fig.savefig(f'../outputs/{filename}_{i}.png', facecolor='k')
