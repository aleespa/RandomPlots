import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt

colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582']
CC = np.random.choice(colors, 60)
m = 0
for j in np.linspace(0, 12 * pi, 210):
    n = 0
    p = plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
    p = plt.axis('off')
    p = plt.xlim(0, 10)
    p = plt.ylim(-10, 10)
    for i in np.linspace(-12, 12, 60):
        p = plt.plot(
            np.linspace(0, 10),
            [sin(x + j) + i for x in np.linspace(0, 10)],
            color=CC[n],
            lw=5,
        )
        n += 1
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/19112019/plor{m}.png',
        facecolor='black',
    )
    m += 1
