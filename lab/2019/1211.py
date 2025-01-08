from math import sqrt

import matplotlib.pylab as plt
import numpy as np

n = 6000
m = 50
p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-sqrt(n), sqrt(n))
p = plt.ylim(-sqrt(n), sqrt(n))
for j in range(m):
    X = np.array([0, 0])
    S = [X]
    for i in range(n):
        X = X + np.random.normal(0, 0.5, 2)
        S.append(X)
    plt.plot(
        [S[i][0] for i in range(n)],
        [S[i][1] for i in range(n)],
        alpha=0.6,
        color=plt.cm.hsv(j / m),
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/11122019.png', facecolor='black')
