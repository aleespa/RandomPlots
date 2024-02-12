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
for z in np.linspace(1,20,100):
    n = 60
    c,s = np.cos(np.linspace(0,2*pi,n)),np.sin(np.linspace(0,2*pi,n))
    r0 = np.random.uniform(1*z,1.2*z)
    r = [r0] + list(np.random.uniform(1*z,1.2*z,n-2)) + [r0]

    plt.plot([c[i]*r[i] for i in range(n)],
             [s[i]*r[i] for i in range(n)],lw=1.1,color=plt.cm.cool(z/20))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/20032020.PNG',facecolor='black')