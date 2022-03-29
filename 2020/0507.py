import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

for i,a in enumerate(np.linspace(0,4,300)):
    p = plt.figure(figsize=(14,14),facecolor='black',dpi=100)
    p = plt.axis('off')
    for t in np.linspace(0,2*pi,150):
        plt.plot([(a**2)*cos(t),cos(8*t)],[(a**2)*sin(t),sin(8*t)],color=plt.cm.RdPu(a/6))

    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/07052020/plot{i}.PNG',facecolor='black')
