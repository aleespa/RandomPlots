
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(12,12),facecolor='black',dpi=800)
p = plt.axis('off')
for z in np.linspace(0,2*pi,15):
    for i in np.linspace(1,20,20):
        plt.plot([i*cos(x+z) for x in np.linspace(0,2*pi,4)],
                 [i*sin(x+z) for x in np.linspace(0,2*pi,4)],
                 color=plt.cm.viridis(i/(20)-0.05),
                 lw=4,alpha=0.85)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/09012020.png',facecolor='black')