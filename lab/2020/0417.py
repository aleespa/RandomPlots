import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for t in range(100):
    plt.plot([cos(x-t) for x in np.linspace(0,2*pi,5)],
             [sin(x+t) for x in np.linspace(0,2*pi,5)],color=plt.cm.Reds(t/100))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17042020/plot{1}.PNG',facecolor='black')

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for t in range(100):
    plt.plot([cos(x-t) for x in np.linspace(0,2*pi,7)],
             [sin(x+t) for x in np.linspace(0,2*pi,7)],color=plt.cm.Blues(t/100))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17042020/plot{2}.PNG',facecolor='black')
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for t in range(100):
    plt.plot([cos(x-t) for x in np.linspace(0,2*pi,9)],
             [sin(x+t) for x in np.linspace(0,2*pi,9)],color=plt.cm.Greens(t/100))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17042020/plot{3}.PNG',facecolor='black')
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for t in range(100):
    plt.plot([cos(x-t) for x in np.linspace(0,2*pi,11)],
             [sin(x+t) for x in np.linspace(0,2*pi,11)],color=plt.cm.Oranges(t/100))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17042020/plot{4}.PNG',facecolor='black')