from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

colors = [plt.cm.YlGn(u) for u in np.random.uniform(0, 1, 50)]
for j, z in enumerate(np.linspace(0, 2 * pi, 210)):
    p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
    p = plt.axis('off')

    for i, t in enumerate(np.linspace(0, 2 * pi, 50)):
        plt.plot(
            [0, cos(t), cos(z)],
            [0, sin(t), sin(z)],
            color=colors[i],
            lw=3.5,
            alpha=0.75,
        )

    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/16042020/plot{j}.PNG',
        facecolor='black',
    )
