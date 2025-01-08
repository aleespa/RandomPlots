import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt


def Random(n):
    x = [0]
    y = [0]
    for i in range(n):
        L = [1, 0, -1]
        s = np.random.choice(L)
        x.append(x[-1] + s)
        if s == 0:
            y.append(y[-1] + np.random.choice([-1, 1]))
        else:
            y.append(y[-1])
    return x, y


p = plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
p = plt.axis('off')
p = plt.xlim(-20, 20.1)
p = plt.ylim(-20, 20.1)
# for x in range(-50,51):
#     p= plt.plot([-x,-x],[-50,50],color='#696969',lw=1,zorder=1)
# for y in range(-50,51):
#     p= plt.plot([-50,50],[-y,-y],color='#696969',lw=1,zorder=1)
p = plt.scatter([0], [0], color='white', zorder=400, s=45)
n = 0
colors = (
    ['#5f865a', '#65a659', '#77bb5d', '#8ed067', '#abdf7b'] * 3
    + ['#ff5e5e', '#ec7f7f', '#ee96be', '#f49ade', '#feb4ff'] * 3
    + ['#3a3663', '#414977', '#476589', '#4c7c9a', '#50919b'] * 3
)
for num in range(45):
    X, Y = Random(400)
    for i in range(1, 9):
        p = plt.plot(
            X[: i * 50], Y[: i * 50], color=colors[num], zorder=3, lw=4, alpha=0.8
        )
        p = p = plt.savefig(
            f'C:/Users/Alejandro/Pictures/RandomPlots/18112019/plor{n}.png',
            facecolor='black',
        )
        n += 1
