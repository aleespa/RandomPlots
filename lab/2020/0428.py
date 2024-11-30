import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm, chi2, binom, gamma
import seaborn as sns
from math import cos, sin, log, tan, pi, exp, sqrt, cosh, sinh, tanh, atan, atan2

p = plt.figure(figsize=(15, 15), facecolor='black', dpi=400)
p = plt.axis('off')
for k in np.linspace(0, 30, 100):
    plt.plot([k * cos(z + pi / 2) * cos(k) for z in np.linspace(0, 2 * pi, 4)],
             [k * sin(z + pi / 2) * sin(k) for z in np.linspace(0, 2 * pi, 4)],
             alpha=0.85,
             lw=np.random.uniform(1, 3.5),
             color=plt.cm.GnBu(k / 30),
             zorder=50 - k)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/28042020.PNG', facecolor='black')
