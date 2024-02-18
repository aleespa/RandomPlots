import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

for i, u in enumerate(list(np.linspace(0,0.9999,120))+list(np.linspace(0,0.9999,120))[::-1]):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    plt.plot([u**2*x+cos(x) for x in np.linspace(0,25*pi,2000)],
             [u**2*x-sin(x) for x in np.linspace(0,25*pi,2000)],
             color=plt.cm.Reds(u**2-0.05),lw=10*u)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/13042020/plot{i}.PNG',facecolor='black')