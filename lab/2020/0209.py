import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

n = 90
R = [0]
for i in range(1,n):
    if R[-1] - i > 0 and (R[-1] - i  not in R):
        R.append(R[-1] - i)
    else:
        R.append(R[-1] + i)
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(0,100)
p = plt.ylim(-50,50)
for i in range(len(R)-1):
    if i%2 == 1:
        plt.plot([0.5*(i+1)*cos(x)+(R[i]+R[i+1])/2 for x in np.linspace(0,pi,50)],
                 [0.5*(i+1)*sin(x)  for x in np.linspace(0,pi,50)],color = 'white',alpha=1,lw=2)
    else:
        plt.fill_between([0.5*(i+1)*cos(x)+(R[i]+R[i+1])/2 for x in np.linspace(pi,2*pi,50)],
                         [0.5*(i+1)*sin(x)  for x in np.linspace(pi,2*pi,50)],color= plt.cm.plasma(i/(n)),alpha=0.1)

plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/09022020.png',facecolor='black')