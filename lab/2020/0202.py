from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

colors = ['#ff729d', '#ffbc77', '#afdf80', '#a4ffcf', '#afdf80'] * 400


def pol(n):
    U = np.linspace(0, 2 * pi, n)
    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    # plt.xlim(-1.05,1.05)
    # plt.ylim(-1.05,1.05)
    for j in range(n):
        for i in range(n):
            if j > i:
                plt.plot(
                    [U[j] * cos(U[j]), U[j] * cos(U[i])],
                    [U[j] * sin(U[j]), U[j] * sin(U[i])],
                    lw=2.5,
                    alpha=0.7,
                    color=colors[i + j],
                )


pol(28)
plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/02022020_2.PNG', facecolor='black'
)
