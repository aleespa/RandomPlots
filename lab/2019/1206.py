import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D

colors = ['#ff9b9b', '#f8ff90', '#a9ff8f', '#22ba5a', '#58c0e7']
p = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
p = plt.axis('off')


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
        color=plt.cm.hsv(a / (1)),
        alpha=0.9,
        lw=1,
    )


for t in np.linspace(0, 2 * pi, 100):
    cubo(cos(t + 0.1) ** 2, sin(t + 0.5) ** 2, pi / 4, 'red')
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/06122019.png', facecolor='black')
