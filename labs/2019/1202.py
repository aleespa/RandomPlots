import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
n=0
for y in np.linspace(0.1,35,210):
    p = plt.figure(figsize=(12,12),facecolor='black')
    p = plt.axis('off')
    plt.plot([cos(x) for x in np.linspace(0,2*pi,4000)],  [cos(x**2)*sin(x) for x in np.linspace(0,y*pi,4000)]  ,lw=0.9,alpha=0.8,color= plt.cm.hsv(n/420))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/02122019/plor{n}.png',facecolor='black')
    n+=1

for y in np.linspace(0.1,35,210):
    p = plt.figure(figsize=(12,12),facecolor='black')
    p = plt.axis('off')
    plt.plot([cos(x) for x in np.linspace(0,2*pi,4000)],  [cos(x**2)*sin(x) for x in np.linspace(0,(35-y)*pi,4000)]  ,lw=0.9,alpha=0.8,color= plt.cm.hsv(n/420))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/02122019/plor{n}.png',facecolor='black')
    n+=1