import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D

def BB(n):
    T = np.linspace(0,1,n)
    B = np.ones(n)*0
    for i in range(n):
        xi = sqrt(2)*np.random.randn()/((i+1)*pi)
        B = B + xi*np.array([sin((i+1)*pi*t) for t in T])
    return B
color1 = ['#96ceb4', '#ffeead','#ff6f69','#ffcc5c','#88d8b0']
color2 = ['#f7f4a3', '#7fccec','#6a81d9','#a479c9','#dfdfdf']
p = plt.figure(figsize=(14,14),facecolor=(0, 0, 0),dpi=400)
p = plt.axis('off')
n = 500
for k in range(15):
    X = BB(n)
    p = plt.plot(np.linspace(0,2*pi,n),
                 [sin(X[i])*cos(i)+k*1.6 for i in range(n)],color =np.random.choice(color1+color2),
                 lw=np.random.uniform(1.5,4),alpha=1)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/07122019.png',facecolor='black')