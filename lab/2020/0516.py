from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np


def arbol(order, theta, sz, posn, heading, colorx=0, line=5):
    trunk_ratio = 0.29
    trunk = sz * trunk_ratio
    delta_x = trunk * cos(heading)
    delta_y = trunk * sin(heading)
    u, v = posn
    newpos = (u + delta_x, v + delta_y)
    plt.plot([u, newpos[0]], [v, newpos[1]], color=plt.cm.YlGn(colorx), lw=line)
    if order > 1:
        newsz = sz * (1 - trunk_ratio)
        arbol(
            order - 1,
            theta,
            newsz,
            newpos,
            heading - theta,
            colorx=colorx + 1 / 10,
            line=line * 0.9,
        )
        arbol(
            order - 1,
            theta,
            newsz,
            newpos,
            heading + theta,
            colorx=colorx + 1 / 10,
            line=line * 0.9,
        )


for i, r in enumerate(np.linspace(0, 2 * pi, 600)):
    plt.figure(figsize=(14, 14), facecolor='black', dpi=200)
    plt.axis('off')
    plt.xlim(-0.8, 0.8)
    plt.ylim(-0.3, 1)
    arbol(order=10, theta=r, sz=1, posn=(0, 0), heading=pi / 2)
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/16052020/plot{i}.PNG',
        facecolor='black',
    )
