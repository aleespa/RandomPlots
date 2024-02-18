import numpy as np
import matplotlib.pylab as plt
from math import pi
import sys
import os

filename = os.path.basename(sys.argv[0])[:-3]

r = np.linspace(0, 2 * pi, 500)

plt.figure(figsize=(10, 10), dpi=400, facecolor='black')
plt.axis('off')
for t in (np.linspace(0.01, 2, 120)**2.5)/(2**2.5):
    c1 = plt.cm.viridis(t / 2+0.2)
    c2 = plt.cm.viridis(t / 2)
    c3 = plt.cm.viridis_r(t / 2 + 0.2)
    c4 = plt.cm.viridis_r(t / 2)
    x = r
    y = np.sin(r * t) * (1 / t)

    plt.plot(x, y, color=c2, alpha=0.7, lw=t-2.5)
    plt.plot(x, -y, color=c1, alpha=0.7, lw=t-2.5)

    plt.plot(4 * pi - x, y, color=c3, alpha=0.7, lw=t-2.5)
    plt.plot(4 * pi - x, -y, color=c4, alpha=0.7, lw=t-2.5)
    plt.xlim(0, 4 * pi)
    plt.ylim(-2 * pi, 2 * pi)

plt.tight_layout()
plt.savefig(f"./../outputs/{filename}.png", facecolor='black',)
# plt.show()
