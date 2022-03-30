import numpy as np
import matplotlib.pylab as plt
from math import pi
import sys
import os
filename = os.path.basename(sys.argv[0])[:-3]

r = np.linspace(0, 2 * pi, 1000)

plt.figure(figsize=(10, 10), dpi=100, facecolor='black')
plt.axis('off')
for t in np.linspace(0,1):
    x = np.cos(t+r)
    y = np.sin(r)
    plt.plot(x, y)
plt.savefig(f"./../outputs/{filename}.png",facecolor='black')
