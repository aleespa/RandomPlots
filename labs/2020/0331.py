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
colors= ['#fe60a1','#c961f7','#ff734c','#3bc7ff','#8089ff']
for u in np.linspace(0,2*pi,20):
    p = plt.plot([2*cos(4*x)*sin(x) for x in np.linspace(0,2*pi,2000)],
                 [2*sin(x)*cos(x)*u for x in np.linspace(0,2*pi,2000)],lw=2,alpha=0.8,zorder=np.random.choice([0,1,2]),
                 color=np.random.choice(colors))
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/31032020.PNG',facecolor='black')