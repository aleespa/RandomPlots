import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D
n=0
for y in np.linspace(0,1,600):
    p = plt.figure(figsize=(12,12),facecolor='black')
    p = plt.axis('off')
    plt.plot([cos(x) for x in np.linspace(0,25*pi,1000)],
             [sin(x) for x in np.linspace(0,25*(1+y)*pi,1000)],lw=4,alpha=1,color=plt.cm.autumn(y))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10122019/plot{n}.png',facecolor='black')
    n+=1