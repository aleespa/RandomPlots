import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D
n=0
for y in np.linspace(-pi,pi,240):
    p = plt.figure(figsize=(12,12),facecolor='black',dpi=200)
    p = plt.axis('off')
    plt.scatter([x*cos(x-y)for x in range(2000)],[x*sin(x) for x in range(2000)],
            s=6,
            color=[plt.cm.autumn(1-x) for x in np.linspace(0,1,2000)],alpha=0.7)
    plt.scatter([x*cos(x+y)for x in range(2000)],[x*sin(x) for x in range(2000)],
            s=6,
            color=[plt.cm.summer(1-x) for x in np.linspace(0,0.9,2000)],alpha=0.7)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/05122019/plot{n}.png',facecolor='black')
    n+=1