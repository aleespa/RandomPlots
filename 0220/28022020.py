import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh


p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in np.linspace(0,1):
    plt.scatter([z**2*x*cos(x*z) for x in np.linspace(0,20*pi,300)],-np.linspace(0,10*pi,300),s=2.5,
                color=plt.cm.cool(z**2),alpha=0.8)
    plt.scatter([z**2*x*cos(x*z) for x in np.linspace(0,20*pi,300)],np.linspace(0,10*pi,300)-63,s=2.5,
                color=plt.cm.cool(1-z**2),alpha=0.8)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/28022020.png',facecolor='black')