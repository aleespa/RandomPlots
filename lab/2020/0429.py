import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

p = plt.figure(figsize=(14,14),facecolor='black',dpi=500)
p = plt.axis('off')
plt.xlim(-12,12)
plt.ylim(-10,10)
for k in np.linspace(0,10,150):
    plt.plot([k*cos(z)*cos(k) for z in np.linspace(0,2*pi,5)],
             [k*sin(z)*sin(k) for z in np.linspace(0,2*pi,5)],
             alpha=0.85,
             lw=np.random.uniform(0.8,3),
             color=plt.cm.rainbow(k/10),
             zorder=50-k)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/29042020.PNG',facecolor='black')