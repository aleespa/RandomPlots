import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog
from mpl_toolkits.mplot3d import Axes3D

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
colors=  ['#64418b','#a1378b','#d42173','#fb294b','#ff5800']
plt.ylim(-0.05,0.45)
for a in np.linspace(1.1,12):
    plt.plot([gamma.pdf(a=a,x=x) for x in np.linspace(1,15,150)],
             lw=np.random.uniform(1,4),
             alpha=0.85,
             color=np.random.choice(colors))
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/28032020.PNG',facecolor='black')