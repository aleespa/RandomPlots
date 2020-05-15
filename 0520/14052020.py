import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2,e
from cmath import exp as cexp, log as clog

plt.figure(figsize=(14,14),dpi=400,facecolor='black')
# plt.xlim(-1.05,1.05)
# plt.ylim(-1.05,1.05)
plt.axis('off')
def pol(n,x,y,color):
    U = np.linspace(0,2*pi,n)
    for j in range(n):
        for i in range(n):
            if i > j:
                plt.plot([cos(U[j])+x, cos(U[i])+x],
                         [sin(U[j])+y, sin(U[i])+y],lw=0.7,color=color)
for k in range(9):
    for l in range(9):
        pol(k+l+4,k*2,l*2,color=plt.cm.GnBu(k*l/(8*8)+0.1))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/14052020.PNG',facecolor='black')