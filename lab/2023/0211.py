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
colors = ['#FFCDD2', '#B2EBF2', '#C8E6C9', '#FFF9C4', '#BBDEFB', '#E1BEE7', '#D7CCC8']

n_points = 200
fig, _ = plt.subplots(figsize=(12, 12), dpi=400)
ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
theta = np.linspace(0, 2 * pi, n_points)
r = np.linspace(0, 1, n_points)
X = np.row_stack([r ** 3 * np.cos(theta), r * np.sin(theta)])
for i, t in enumerate(np.linspace(0, 2 * pi, 32)):
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t), np.cos(t)]])
    ax.scatter((R @ X)[0, :], (R @ X)[1, :], lw=0, alpha=0.8, color=colors[i % len(colors)],
               s=np.linspace(3, 30, n_points))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

fig.savefig(f'../outputs/{filename}.png', facecolor='k')
