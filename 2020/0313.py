import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh

for r,i in zip(np.linspace(2,-2,240),range(240)):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    p = plt.xlim(-2,2)
    p = plt.ylim(-2,2)
    for u in np.linspace(0,2*pi,20):
        for z in np.linspace(0,2*pi,20):
            plt.plot([r*cos(u),r*cos(u)+cos(z)],[r*sin(u),r*sin(u)+sin(z)],color=plt.cm.Spectral(z/(2*pi)),alpha=0.9,lw=2.3)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/13032020/plot{i}.PNG',facecolor='black')
