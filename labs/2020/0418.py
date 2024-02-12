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
for u in np.linspace(0,1,40):
    if u <=0.5:
        plt.plot([0,0,1-u,1-u,0],[1-u,0,0,1-u,1-u],lw=3,color=plt.cm.hsv(1-u*2))
    plt.plot([0,0,u,u,0],[1-u,0,0,1-u,1-u],lw=3,color=plt.cm.hsv(1-u))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/18042020.PNG',facecolor='black')