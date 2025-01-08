
from math import cos, sin, log, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
def drawCircle(x,y,r):
    plt.plot([r*cos(t)+x for t in np.linspace(0,2*pi,120)],
             [r*sin(t)+y for t in np.linspace(0,2*pi,120)],lw=r**0.2,color=plt.cm.viridis(log(r)/log(2**8)),zorder=int(r))
    if r>=1:
        drawCircle(x+r/2 , y+r/2, r/2)
        drawCircle(x-r/2 , y-r/2, r/2)
#     print(r)
drawCircle(0,0,2**8)
def drawCircle(x,y,r):
    plt.plot([r*cos(t)+x for t in np.linspace(0,2*pi,120)],
             [r*sin(t)+y for t in np.linspace(0,2*pi,120)],lw=r**0.2,color=plt.cm.viridis(log(r)/log(2**8)),zorder=int(r))
    if r>=1:
        drawCircle(x-r/2 , y+r/2, r/2)
        drawCircle(x+r/2 , y-r/2, r/2)
#     print(r)
drawCircle(0,0,2**8)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/08052020.PNG',facecolor='black')