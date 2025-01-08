import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=600)
p = plt.axis('off')
p = plt.xlim(-280, 280)
p = plt.ylim(-280, 280)


def cubo(a, b, t, color):
    p = plt.plot(
        [
            a * cos(t) - a * sin(t),
            b * cos(t) - a * sin(t),
            b * cos(t) - b * sin(t),
            a * cos(t) - b * sin(t),
            a * cos(t) - a * sin(t),
        ],
        [
            a * sin(t) + a * cos(t),
            b * sin(t) + a * cos(t),
            b * sin(t) + b * cos(t),
            a * sin(t) + b * cos(t),
            a * sin(t) + a * cos(t),
        ],
        color=plt.cm.PuRd(a / (20 * pi) - 0.3),
        alpha=1,
        lw=1.8,
    )


for t in np.linspace(0, 20 * pi, 120):
    cubo(t, 3 * t + 1, t * 4.465114832535881, 'red')
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/03122019.png', facecolor='black')
