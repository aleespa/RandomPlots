from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np


def rama(x, y, t, c):
    plt.plot([x, x + cos(t)], [y, y + sin(t)], color=plt.cm.cool(c), lw=1.2)
    plt.plot([x, x + cos(pi - t)], [y, y + sin(pi - t)], color=plt.cm.spring(c), lw=1.2)


def rama2(x, y, t, c):
    plt.plot([x, x - cos(t)], [y, y - sin(t)], color=plt.cm.cool(c), lw=1.2)
    plt.plot([x, x - cos(pi - t)], [y, y - sin(pi - t)], color=plt.cm.summer(c), lw=1.2)


def branch(X, t, c):
    Z = []
    for z in X:
        K = 2
        rama(z[0], z[1], t, c)
        Z.append([z[0] + cos(t), z[1] + sin(t)])
        Z.append([z[0] + cos(pi - t), z[1] + sin(pi - t)])
    return (np.array(Z))


def branch2(X, t, c):
    Z = []
    for z in X:
        K = 2
        rama2(z[0], z[1], t, c)
        Z.append([z[0] - cos(t), z[1] - sin(t)])
        Z.append([z[0] - cos(pi - t), z[1] - sin(pi - t)])
    return (np.array(Z))


fig, ax = plt.subplots(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
t = pi / 3
X = branch(np.array([[0, -18]]), t, 0)
for c in np.linspace(0.1, 1, 9):
    X = branch(X, t, c)
    t *= 1.01

t = pi / 3
X = branch2(np.array([[0, 0]]), t, 0)
for c in np.linspace(0.1, 1, 9):
    X = branch2(X, t, c)
    t *= 1.01

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10052020.PNG', facecolor='black')
