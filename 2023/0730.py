import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy import sqrt, pi, cos, sin
import pandas as pd
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]

T = np.linspace(0, 2 * pi, 2000)
for n, epsilon in enumerate(np.linspace(0, 2 * pi, 750)):
    if n % 3 == 2:
        fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
        for k in range(20):
            T_sector = T[k * 100: (k+1)*100 +1] + epsilon
            T_0 = T_sector + epsilon + epsilon

            R_1 = 5 + 1.5 * sin(40 * T_0) + 0.5 * cos(10 * T_0) + 0.1 * sin(50 * T_0) ** 2 * cos(10 * T_0) + 0.1
            R_2 = 5 + 1.5 * cos(40 * T_0) + 0.5 * sin(10 * T_0) + 0.1 * cos(50 * T_0) ** 2 * sin(10 * T_0) + 0.1
            ax.plot(cos(T_sector) * R_1,
                    sin(T_sector) * R_2, color=plt.cm.pink(k/20), lw=(k * 6)/20)
            if k == 19:
                ax.scatter(cos(T_sector[-1]) * R_1[-1],
                           sin(T_sector[-1]) * R_2[-1], color='w', s=100)

        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)

        fig.savefig(f'../outputs/20230730_v4/{filename}_{n}.png', facecolor='k')
        plt.close()
