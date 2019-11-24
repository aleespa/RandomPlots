import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

def BB(n):
    T = np.linspace(0,1,n)
    B = np.ones(n)*0
    for i in range(n):
        xi = sqrt(2)*np.random.randn()/((i+1)*pi)
        B = B + xi*np.array([sin((i+1)*pi*t) for t in T])
    return B
colors = ['#daf8e3','#97ebdb','#00c2c7','#0086ad','#005582']
p = plt.figure(figsize=(12,12),facecolor='black',dpi=200)
p = plt.axis('off')
p = plt.xlim(-0.1,1.1)
p = plt.ylim(-1.5,1.5)
p = plt.scatter([0,1],[0,0],s=40,color='red',zorder=200)
for i in range(120):
    p= plt.plot(np.linspace(0,1,2000),BB(2000),color=np.random.choice(colors),alpha=0.6)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17112019/plor{i}.png',facecolor='black')