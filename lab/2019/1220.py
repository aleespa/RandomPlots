import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt

m = 0
lws = np.random.normal(7, 1, 50)
cols = np.random.uniform(0, 1, 50)
for b in np.linspace(0, 2 * pi, 300):
    p = plt.figure(figsize=(13, 13), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-2.1, 2.1)
    p = plt.ylim(-2.1, 2.1)
    i = 0
    for z in np.linspace(1, 7, 50):
        plt.plot(
            [
                cos(t) * z * (1 - cos(t * 5) * sin(2 * t))
                for t in np.linspace(0, 2 * pi, 500)
            ],
            [
                sin(t) * z * (1 - cos(t * 5) * sin(b + 2 * t))
                for t in np.linspace(0, 2 * pi, 500)
            ],
            lw=lws[i],
            alpha=0.75,
            color=plt.cm.Blues(cols[i]),
        )
        i += 1
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/20122019/plot{m}.png',
        facecolor='black',
    )
    m += 1
