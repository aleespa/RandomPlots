import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

colors = matplotlib.cm.get_cmap('rainbow')
X, Y = [log(t) * cos(t * 1.1) for t in np.linspace(10, 60 * pi, 4200)], [
    log(t) * sin(t) for t in np.linspace(10, 50 * pi, 4200)
]

n = 0
for i in range(208):
    p = plt.figure(figsize=(12, 12), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-5.3, 5.3)
    p = plt.ylim(-5.3, 5.3)
    p = plt.plot(
        X[20 * i : 20 * (i + 2)],
        Y[20 * i : 20 * (i + 2)],
        alpha=1,
        lw=3,
        color=colors(i / 210),
    )
    p = plt.scatter([X[20 * (i + 2)]], [Y[20 * (i + 2)]], color='white')
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/25112019/plor{n}.png',
        facecolor='black',
    )
    n += 1
p = plt.figure(figsize=(12, 12), facecolor='black')
p = plt.axis('off')
p = plt.xlim(-5.3, 5.3)
p = plt.ylim(-5.3, 5.3)

for i in range(208):
    p = plt.plot(
        X[20 * i : 20 * (i + 2)],
        Y[20 * i : 20 * (i + 2)],
        alpha=0.9,
        lw=2.5,
        color=colors(i / 210),
    )
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/25112019/plor{n}.png',
        facecolor='black',
    )
    n += 1
