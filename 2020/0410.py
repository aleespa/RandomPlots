import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog

p = plt.figure(figsize=(14,14),facecolor='black',dpi=100)
p = plt.axis('off')
p = plt.ylim(-2,2)
p = plt.plot(np.linspace(0,2*pi),[sin(x) for x in np.linspace(0,2*pi)],color='red',lw=4)
for i,z in enumerate(np.linspace(0,2*pi,200)):
    p = plt.plot(np.linspace(0,2*pi),[sin(z) + cos(z)*(x-z) for x in np.linspace(0,2*pi)],color=plt.cm.summer(z/(2*pi)))
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10042020/plot{i}.PNG',facecolor='black',lw=3)

