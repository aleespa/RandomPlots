import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
n = 80
p = plt.xlim(-5,5)
p = plt.ylim(-5,5)
for i in range(n):
    r = np.random.uniform(2,5)
    c = np.random.uniform(0,1)
    plt.plot([cos(x)*r for x in np.linspace(i*(2*pi)/n,(i+1)*(2*pi)/n,200)],
             [sin(x)*r for x in np.linspace(i*(2*pi)/n,(i+1)*(2*pi)/n,200)],color=plt.cm.autumn(1-c),lw=2)
    plt.plot([cos(i*(2*pi)/n)*r,0],
             [sin(i*(2*pi)/n)*r,0],color=plt.cm.winter(c),lw=2)
    plt.plot([cos((i+1)*(2*pi)/n)*r,0],
             [sin((i+1)*(2*pi)/n)*r,0],color=plt.cm.winter(c),lw=2)


plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/19032020.PNG',facecolor='black')