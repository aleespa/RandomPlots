import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D


def Lolka(VI):
    alpha, beta, gamma, delta = -0.4, 0.01, 0.2, -0.01
    h = 0.01
    f = lambda x: np.array(
        [
            alpha * x[0] + beta * x[0] * x[1] + np.random.uniform(-3, 3),
            gamma * x[1] + delta * x[0] * x[1],
        ]
    )
    U = [np.array(VI)]
    T = np.arange(0, 400, h)
    for j in range(len(T) - 1):
        k1 = f(U[j])
        k2 = f(U[j] + (h / 2) * k1)
        k3 = f(U[j] + (h / 2) * k2)
        k4 = f(U[j] + h * k3)
        U.append(U[j] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    p = plt.plot(
        [U[i][0] for i in range(len(U))],
        [U[i][1] for i in range(len(U))],
        alpha=0.8,
        lw=1,
    )


p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
for i in range(1, 11):
    Lolka([5 * i, 80])
p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/23112019.png', facecolor='black'
)
