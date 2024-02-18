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
for z in np.linspace(-8,8,500) :
    plt.plot([tanh(x*z) for x in np.linspace(-pi,pi,150)],lw=np.random.uniform(0.5,4)
             ,alpha=0.75,color=plt.cm.winter(np.random.uniform(0.2,1)))
plt.plot([tanh(x) for x in np.linspace(-pi,pi,150)],lw=5,alpha=0.8,color='red')
plt.plot([tanh(-x) for x in np.linspace(-pi,pi,150)],lw=5,alpha=0.8,color='red')

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/03032020.PNG',facecolor='black')