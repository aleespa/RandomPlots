import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2



p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in np.linspace(0,5,50):
    plt.plot([cos(t+z)*z for t in np.linspace(0,2*pi,4)],
             [sin(t+z)*z for t in np.linspace(0,2*pi,4)],
             lw=3
             ,
             color=plt.cm.Spectral(z/5))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/02052020.PNG',facecolor='black')