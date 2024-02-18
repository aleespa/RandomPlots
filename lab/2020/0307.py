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
for z in np.linspace(0,1,50):
    plt.plot([z*cos(x)*sin(x) for x in np.linspace(1,3*pi,500)],
             [z*cos(x)*sin(x*2) for x in np.linspace(1,3*pi,500)]
             ,alpha=0.9,color=plt.cm.RdPu(z),lw=2)
    
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/07032020.PNG',facecolor='black')