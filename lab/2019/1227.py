import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin, log, tan, gamma, pi, exp, sqrt

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=1000)
p = plt.axis('off')
for z in range(30):
    xx, yy = np.meshgrid(np.linspace(0, 1, z), np.linspace(0, 1, z))
    plt.plot(xx, yy, marker='.', linestyle='none')
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/27122019.png', facecolor='black')
