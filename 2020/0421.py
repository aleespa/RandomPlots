import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for t in range(110):
    plt.plot([cos(x-t)*t for x in np.linspace(0,2*pi,5)],
             [sin(x)*t for x in np.linspace(0,2*pi,5)],color=plt.cm.BuPu(t/100),zorder=120-t,alpha=0.85,lw=3)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/21042020.PNG',facecolor='black')