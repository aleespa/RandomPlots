import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import os
import sys
import itertools
filename = os.path.basename(sys.argv[0])[:-3]

B = -1
C = 0.5

# np.random.seed(2)
plt.figure(num=1, clear=True, figsize=(12, 12), dpi=500, facecolor='black')
plt.axis('off')
nloops = 400
n_points = 4000
for eloop in range(0, nloops):

    xlast = np.random.normal(0, 1, 1)
    ylast = np.random.normal(0, 1, 1)

    xnew = np.zeros(shape=(n_points,))
    ynew = np.zeros(shape=(n_points,))
    for loop in range(0, n_points):
        xnew[loop] = 1 + ylast - C * abs(xlast)
        ynew[loop] = B * xlast
        xlast = xnew[loop]
        ylast = ynew[loop]

    plt.scatter(np.real(xnew), np.real(ynew), s=0.3, color=plt.cm.plasma(eloop/nloops), alpha=0.9, lw=0)
    plt.xlim(-1.35, 2.1)
    plt.ylim(-2.1, 1.35)

plt.savefig(f"./../outputs/{filename}.png", facecolor='black')