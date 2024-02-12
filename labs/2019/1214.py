import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D
n= 800
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-10.1,10.1)
p = plt.ylim(-10.1,10.1)
X,Y = np.random.uniform(-10,10,n),np.random.uniform(-10,10,n)
plt.scatter(X,Y,color='white',zorder=n**2,alpha=0.7,s=8)
for i in range(n):
    plt.plot([X[i],X[i]*0.7],[Y[i],Y[i]*1.5],alpha=0.8,lw=np.random.uniform(1,5))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/14122019.png',facecolor='black')