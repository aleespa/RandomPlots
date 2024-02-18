import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

for i,a in enumerate(np.linspace(1,2,360)):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    for t in np.linspace(0,20*pi,180):
        plt.plot([2*cos(t),cos(a*t)],[2*sin(t),sin(a*t)],color='white')
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/06052020/plot{i}.PNG',facecolor='black')