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
colors = ['#ffe7f8','#ffcfee','#c8bbff','#bff0ff','#dff8ff']
for u in np.linspace(0,2*pi,100):
    for z in np.linspace(0,2*pi,9):
        plt.plot([cos(u),cos(u)+cos(z)],[sin(u),sin(u)+sin(z)],alpha=1,lw=2,
                 color=np.random.choice(colors),zorder=np.random.randint(50))

plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/18032020.PNG',facecolor='black')