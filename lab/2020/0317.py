from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=500)
p = plt.axis('off')
plt.xlim(-50, 50)
plt.ylim(-50, 50)
for t in np.linspace(0, pi):
    plt.scatter(
        [x * cos(x + t) for x in np.linspace(0, 50, 1000)],
        [x * sin(x) for x in np.linspace(0, 51, 1000)],
        s=4,
        color=plt.cm.YlGnBu(t / pi),
        alpha=0.7,
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17032020.PNG', facecolor='black')
