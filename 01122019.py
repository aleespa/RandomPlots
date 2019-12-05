import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
# p = plt.xlim(-.5,.5)
# p = plt.ylim(0,1)
for x in np.linspace(-2*pi,2*pi,300):
    plt.plot([-x,x],[cos(-x),sin(x+5)],lw=0.9,color=plt.cm.Spectral((x+2*pi)/(4*pi)),alpha=0.8)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/01122019.png',facecolor='black')