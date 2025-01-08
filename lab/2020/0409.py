from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

for i, u in enumerate(np.linspace(0, 2 * pi, 210)):
    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=200)
    p = plt.axis('off')
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    for m in range(-4, 4):
        for k in range(-4, 4):
            if k > m:
                plt.plot([k, cos(0 + u) + k], [m, sin(0 + u) + m], lw=6, alpha=0.9)
                plt.plot(
                    [k, cos(1 / 3 * (2 * pi) + u) + k],
                    [m, sin(1 / 3 * (2 * pi) + u) + m],
                    lw=6,
                    alpha=0.9,
                )
                plt.plot(
                    [k, cos(2 / 3 * (2 * pi) + u) + k],
                    [m, sin(2 / 3 * (2 * pi) + u) + m],
                    lw=6,
                    alpha=0.9,
                )
            else:
                plt.plot([k, cos(0 - u) + k], [m, sin(0 - u) + m], lw=6, alpha=0.9)
                plt.plot(
                    [k, cos(1 / 3 * (2 * pi) - u) + k],
                    [m, sin(1 / 3 * (2 * pi) - u) + m],
                    lw=6,
                    alpha=0.9,
                )
                plt.plot(
                    [k, cos(2 / 3 * (2 * pi) - u) + k],
                    [m, sin(2 / 3 * (2 * pi) - u) + m],
                    lw=6,
                    alpha=0.9,
                )
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/09042020/plot{i}.PNG',
        facecolor='black',
    )
