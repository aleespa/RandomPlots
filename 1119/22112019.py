
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D

for i in range(360):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d',facecolor='black')
    ax.set_zlim(-0.2,0.7)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.view_init(i, 0)
    p = plt.axis('off')
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sin(xx**2 + yy**2) / ((xx)**2 + yy**2)
    ax = ax.plot_wireframe(xx, yy, z, rstride=1, cstride=1,alpha=0.8,lw=0.8,color= (1-i/360,i**2/(360**2),i/360))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/22112019/plor{i}.png',facecolor='black')