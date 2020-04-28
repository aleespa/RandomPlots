import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog



p = plt.figure(figsize=(16,16),facecolor='black',dpi=400)
p = plt.axis('off')
for k in np.linspace(0,30,90):
    plt.plot([k*cos(z)+cos(k) for z in np.linspace(0,2*pi,5)],
             [k*sin(z)+sin(k) for z in np.linspace(0,2*pi,5)],
             alpha=0.8,
             lw=np.random.uniform(0.8,3),
             color=plt.cm.Spectral(((k+25)/50)))
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/27'
                f'042020.PNG',facecolor='black')