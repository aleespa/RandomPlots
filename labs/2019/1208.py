import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-400,400)
p = plt.ylim(-400,400)
for y in np.linspace(0,2*pi,28):
    plt.scatter([x*cos(x) + 200*cos(y) for x in np.linspace(0,200,500)],
                [x*sin(x) + 200*sin(y) for x in np.linspace(0,200,500)],
                s=[(x) for x in np.linspace(7,4,500)],
                color=[plt.cm.cool(x) for x in np.linspace(0,0.5,500)],
                alpha=1)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/08122019.png',facecolor='black')