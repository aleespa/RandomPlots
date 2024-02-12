import numpy as np
import matplotlib.pylab as plt
from math import sqrt, pi
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]

colors = ['#19d9b4', '#92d275', '#cccccc', '#bbbbbb', '#aaaaaa']
C = np.random.choice(colors, 50)

plt.figure(num=1, clear=True, figsize=(14, 14), dpi=400, facecolor='black')
p = plt.axis('off')
u = 0.1
X = np.linspace(0, pi, 100)
n = 20
for i, z in enumerate(np.linspace(0, 2 * pi, n)):
    plt.plot(X, np.cos(X * z) + z / u,
             lw=3.5, color=plt.cm.viridis(i / n))
    plt.plot(-X, np.cos(X * z) + z / u,
             lw=3.5, color=plt.cm.viridis(i / n))

    plt.plot(X, -np.cos(X * z) - z / u,
             lw=3.5, color=plt.cm.viridis(i / n))

    plt.plot(-X, -np.cos(X * z) - z / u,
             lw=3.5, color=plt.cm.viridis(i / n))
    print(z/u)

plt.savefig(f"./../outputs/{filename}.png", facecolor='black', )
