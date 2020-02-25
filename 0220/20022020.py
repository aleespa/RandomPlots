import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
n =5
for z in np.linspace(0,2*pi,80):
    plt.plot([cos(x)*cos(z*2) for x in np.linspace(0,2*pi,n)],
             [sin(x)*sin(z*2) for x in np.linspace(0,2*pi,n)],lw=2,alpha=.8)

plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/20022020.png',facecolor='black')