import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt
from mpl_toolkits.mplot3d import Axes3D

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-1.1, 1.1)
p = plt.ylim(-1.1, 1.1)
for r in [1, 0.5, 0.25]:
    for t in np.linspace(0, pi, 70):
        # plt.scatter([r*cos(0+t),r*cos((2/3)*pi+t),r*cos(4*pi/3+t),r*cos((2)*pi+t)],
        #         [r*sin(0+t),r*sin((2/3)*pi+t),r*sin(4*pi/3+t),r*sin((2)*pi+t)],color='white',zorder=200,alpha=0.8,s=4)
        plt.plot(
            [
                r * cos(0 + t),
                r * cos((2 / 3) * pi + t),
                r * cos(4 * pi / 3 + t),
                r * cos((2) * pi + t),
            ],
            [
                r * sin(0 + t),
                r * sin((2 / 3) * pi + t),
                r * sin(4 * pi / 3 + t),
                r * sin((2) * pi + t),
            ],
            color=plt.cm.hsv(((t) / (pi))),
            alpha=0.8,
            lw=(r + 0.1),
        )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/30112019.png', facecolor='black')
