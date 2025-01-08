import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D

X, Y = [x * cos(x) for x in range(30000)], [x * sin(x**1.0001) for x in range(30000)]
t = np.linspace(3000, 30000, 300)
for i in range(50, 300):
    p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    p = plt.axis('off')
    p = plt.xlim(-t[i] - 500, t[i] + 500)
    p = plt.ylim(-t[i] - 500, t[i] + 500)
    plt.scatter(
        X,
        Y,
        s=[x**4 for x in np.linspace(0.5, 1.4, 30000)],
        color=[plt.cm.autumn(x) for x in np.linspace(0, 1, 30000)],
    )
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/04122019/plot{i}.png',
        facecolor='black',
    )
