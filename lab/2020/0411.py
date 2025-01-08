from math import sqrt

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.ylim(-2, 2)
p = plt.xlim(-2, 2)
# plt.plot([cos(x) for x in np.linspace(0,2*pi,100)],[sin(x) for x in np.linspace(0,2*pi,100)],color='red',lw=3)
for a in np.linspace(-0.999, 0.999, 60):
    p = plt.plot(
        np.linspace(-2, 2),
        [sqrt(1 - a**2) - a / (sqrt(1 - a**2)) * (x - a) for x in np.linspace(-2, 2)],
        color=plt.cm.spring((a + 1) / (2)),
        lw=2,
    )
    p = plt.plot(
        np.linspace(-2, 2),
        [-sqrt(1 - a**2) + a / (sqrt(1 - a**2)) * (x - a) for x in np.linspace(-2, 2)],
        color=plt.cm.autumn((a + 1) / (2)),
        lw=2,
    )
p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/11042020.PNG', facecolor='black'
)
