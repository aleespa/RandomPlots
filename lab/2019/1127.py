from math import cos, sin, pi

import matplotlib
import matplotlib.pylab as plt
import numpy as np

colors = matplotlib.cm.get_cmap('viridis')
n = 0

p = plt.figure(figsize=(15, 15), facecolor='black', dpi=400)
p = plt.axis('off')
s = 0.5
p = plt.xlim(-1.1, 1.1)
p = plt.ylim(-1.1, 1.1)

for i in np.linspace(-pi, pi, 80):
    X = np.array([cos(t) + sin(i) for t in np.linspace(0, 2 * pi, 400)])
    Y = np.array([sin(t) + cos(i) for t in np.linspace(0, 2 * pi, 400)])
    plt.plot(
        X,
        Y,
        lw=np.random.choice([1, 1, 0.5, 0.7, 3, 4, 2]),
        alpha=np.random.uniform(0.5, 1),
        color=colors(np.random.uniform(0, 1)),
    )

X = np.array([0.45 * cos(t) for t in np.linspace(0, 2 * pi, 400)])
Y = np.array([0.45 * sin(t) for t in np.linspace(0, 2 * pi, 400)])
plt.plot(X, Y, lw=25, alpha=0.8, color='black', zorder=100)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/27112019.png', facecolor='black')
