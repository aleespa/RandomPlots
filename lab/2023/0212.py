import os
import sys

import matplotlib.pylab as plt
import numpy as np
from numpy import pi, cos

filename = os.path.basename(sys.argv[0])[:-3]
colors = [
    '#FF69B4',
    '#DC143C',
    '#FFC0CB',
    '#FFB6C1',
    '#DB7093',
    '#FFA07A',
    '#F08080',
    '#FFDAB9',
]


n_points = 800
fig, _ = plt.subplots(figsize=(12, 12), dpi=400)
ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='black')
theta = np.linspace(0, 2 * pi, n_points)
r = (1 - cos(theta)) ** 2 + 5
X = np.row_stack([r * np.cos(theta), r * np.sin(theta * 6)])
s = np.linspace(1, 20, n_points)
for i, t in enumerate(np.linspace(0, 2 * pi, 16)):
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ax.scatter(
        (R @ X)[0, :],
        (R @ X)[1, :],
        lw=0,
        alpha=0.75,
        color=colors[i % len(colors)],
        s=s,
    )

# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)

fig.savefig(f'../outputs/{filename}.png', facecolor='k')
