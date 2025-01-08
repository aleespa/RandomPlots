import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt

colors = ['#aaaaaa', '#e37398', '#f0c1cc', '#fee2e9', '#fe7ea8']

p = plt.figure(figsize=(14, 14), facecolor='black')
p = plt.axis('off')
# p = plt.xlim(-2.5,2)
# p = plt.ylim(-2,2)
for y in np.linspace(0, 2 * pi, 18):
    plt.plot(
        [cos(x) + cos(y) * 3 for x in np.linspace(0, 2 * pi, int(y * 2 * pi))],
        [sin(x) + sin(y) * 3 for x in np.linspace(0, 2 * pi, int(y * 2 * pi))],
        lw=5,
        alpha=0.9,
        color=np.random.choice(colors),
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/09122019.png', facecolor='black')
