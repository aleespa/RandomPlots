from math import sqrt

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
N = 14000
X, Y = np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N)
color = [sqrt(X[i] ** 2 + Y[i] ** 2) * 0.9 for i in range(N)]
plt.scatter(X, Y, color=plt.cm.binary(color), s=8)

for z in np.linspace(0.2, 1):
    b = [
        sqrt(X[i] ** 2 + Y[i] ** 2) < 0.8 * z and sqrt(X[i] ** 2 + Y[i] ** 2) > 0.74 * z
        for i in range(N)
    ]
    plt.scatter(X[b], Y[b], s=10, color=plt.cm.binary(z))

b = [
    sqrt(X[i] ** 2 + Y[i] ** 2) < 0.8 and sqrt(X[i] ** 2 + Y[i] ** 2) > 0.75
    for i in range(N)
]
plt.scatter(X[b], Y[b], s=10, color=plt.cm.spring(np.random.uniform(0, 1, sum(b))))
p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/04032020.PNG', facecolor='black'
)
