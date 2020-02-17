import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for l in np.linspace(0,2*pi,100):
    p = plt.plot([sqrt((cos(l)-1)**2 +(sin(l)-1)**2)*cos(z) +cos(l)-1 for z in np.linspace(0,2*pi)],
                 [sqrt((cos(l)-1)**2 +(sin(l)-1)**2)*sin(z) + sin(l)-1 for z in np.linspace(0,2*pi)],
                 lw=np.random.uniform(0.5,5),
                 alpha=0.7,
                 color=plt.cm.RdPu(np.random.uniform(0,1)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/14022020.png',facecolor='black')