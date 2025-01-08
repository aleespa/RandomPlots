from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

h = np.random.choice([-1, 0, 1], 10)
f = np.random.binomial(10, 0.5, 10)
X = np.linspace(0, 4 * pi, 100)
colors = np.random.choice(['#4a2356', '#f6e9f8', '#f9f056', '#f4d647', '#9752f5'], 30)
for k, t in enumerate(np.linspace(0, 2 * pi, 180)):
    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
    p = plt.axis('off')
    plt.xlim(-22, 22)
    plt.ylim(-22, 22)
    for j, z in enumerate(np.linspace(0, 1, 30)):
        plt.scatter(
            [
                z * cos(x) * (sum([h[i] * cos(f[i] * x + t) for i in range(10)]) + 15)
                for x in X
            ],
            [
                z * sin(x) * (sum([h[i] * sin(f[i] * x + t) for i in range(10)]) + 15)
                for x in X
            ],
            color=colors[j],
            s=32,
            lw=0.8,
        )
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/25042020/plot{k}.PNG',
        facecolor='black',
    )
