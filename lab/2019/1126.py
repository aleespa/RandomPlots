import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

colors = matplotlib.cm.get_cmap('summer')
n = 0
for s in np.linspace(0.5, 15, 210):
    p = plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
    p = plt.axis('off')
    p = plt.xlim(-1, 1)
    p = plt.ylim(-1, 1)
    X = np.array([t**s * cos(t) for t in np.linspace(0, 40 * pi, 4000)])
    Y = np.array([t**s * sin(t) for t in np.linspace(0, 40 * pi, 4000)])
    h = (40 * pi) ** s
    p = plt.plot(X / h, Y / h, lw=3, color=colors(s / 15))
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/26112019/plor{n}.png',
        facecolor='black',
    )
    n += 1
