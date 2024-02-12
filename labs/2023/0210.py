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
colors = ["#1B2331", "#34495E", "#2C3E50", "#16A085", "#27AE60", "#2980B9", "#8E44AD", "#2C3E50", "#F1C40F", "#F7DC6F",
          "#E67E22", "#E74C3C", "#9B59B6", "#FB6964", "#3498DB", "#1ABC9C", "#2ECC71", "#9B59B6", "#F1C40F", "#F7DC6F",
          "#E67E22"]

fig, _ = plt.subplots(figsize=(12, 12), dpi=400)
ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
for i in range(380):
    r, theta1, theta2 = np.random.uniform(0, 0.9, 3)
    theta = np.linspace(theta1, theta2, 70) * 4 * pi - 2 * pi
    plt.scatter(r * np.cos(theta), r * np.sin(theta) * r * r, color=colors[i % len(colors)], s=5, alpha=0.8, lw=0)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

fig.savefig(f'../outputs/{filename}.png', facecolor='k')
