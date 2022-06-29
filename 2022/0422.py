import numpy as np
import matplotlib.pylab as plt
from math import sqrt
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys

filename = os.path.basename(sys.argv[0])[:-3]


def downsample(x, y, n):
    interpolated = interp1d(x, y, axis=0, fill_value='extrapolate', kind='nearest')
    downsampled = interpolated(np.linspace(x[0], x[-1], n))
    return downsampled


def brownian(x0, n, dt, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out


frames = 1000
n = 2000
m = 22
dt = 0.01
T = n * dt
X = np.linspace(0, T, 2 * n)
Y1 = brownian(0, n, dt)
Y2 = brownian(0, n, dt)
Y3 = brownian(0, n, dt)

for j in range(frames):
    Y1= downsample(np.linspace(0, X[-1] + m * dt, n + m), np.concatenate([Y1, brownian(Y1[-1], m, dt)]), n)
    X = np.linspace(0, X[-1] + m * dt, n)

    plt.figure(num=1, clear=True, figsize=(12, 12), dpi=200, facecolor='black')
    plt.axis('off')
    plt.fill_between(X[1:], Y1[1:], -Y1[1:], alpha=0.5, facecolor=plt.cm.Spectral(j/frames), edgecolor='white')
    plt.fill_between(-X[::-1], Y1[::-1], -Y1[::-1], alpha=0.5, facecolor=plt.cm.Spectral(j/frames), edgecolor='white')
    plt.ylim(-sqrt(X[-1]) * 2, sqrt(X[-1]) * 2)
    plt.tight_layout()
    plt.savefig(f"./../outputs/{filename}/plot{frames - j}.png", facecolor='black', )
