import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm
import seaborn as sns
from math import cos, sin, log, tan, gamma, pi, exp, sqrt, cosh, sinh, tanh

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
X, Y = [cos(x) / (sin(x * 6) + 2) * x for x in np.linspace(0, 60 * pi, 10000)], [sin(x) / (cos(x * 6) + 2) * x for x in
                                                                                 np.linspace(0, 60 * pi, 10000)]
for j in range(99):
    plt.scatter(X[j * 100:(j + 1) * 100 + 1], Y[j * 100:(j + 1) * 100 + 1], s=6, alpha=0.9,
                color=[plt.cm.spring(i) * (j % 8 in [0, 1]) +
                       plt.cm.summer(i) * (j % 8 in [2, 3]) +
                       plt.cm.autumn(i) * (j % 8 in [4, 5]) +
                       plt.cm.winter(i) * (j % 8 in [6, 7]) for i in np.linspace(0, 1, 101)])
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/05032020.PNG', facecolor='black')
