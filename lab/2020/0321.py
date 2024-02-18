import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog



for u,i in zip(np.linspace(0,12,450),range(450)):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    for z in np.linspace(0,2,50):
        plt.plot(np.linspace(z,2*pi-z,300),[0.5*sin(x+z*u) +z for x in np.linspace(0,2*pi,300)],lw=2.5,color=plt.cm.spring(z/2))
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/21032020/plot{i}.PNG',facecolor='black')