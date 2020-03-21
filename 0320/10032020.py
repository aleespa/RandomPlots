import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for u in np.linspace(0,2*pi,40):
    for z in np.linspace(0,2*pi,40):
        plt.plot([cos(u),cos(u)+cos(z)],[sin(u),sin(u)+sin(z)],color=plt.cm.hsv(z/(2*pi)),alpha=0.7,lw=1.3)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10032020.png',facecolor='black')