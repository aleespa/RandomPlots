import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for r in np.linspace(0,2*pi,200):
    X = [0]
    Y = [0]
    for _ in range(1000):
        X.append(X[-1] + np.random.normal(cos(r),2))
        Y.append(Y[-1] + np.random.normal(sin(r),2))
    plt.plot(X,Y,alpha=0.65,lw=0.8,color=plt.cm.hsv(r/(2*pi)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/06012020.png',facecolor='black')