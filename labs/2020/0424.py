import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog
from mpl_toolkits.mplot3d import Axes3D

n = 20
h = np.random.choice([-1,0,1],(10,n))
f = np.random.binomial(10,0.5,(10,n))

X = np.linspace(0,4*pi,500)
colors= np.random.choice(['#f00000', '#ff0074', '#ff8100','#ffc100' ,'#f0ff00','white'],n)

for k,t in enumerate(np.linspace(0,2*pi,200)):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    plt.xlim(-22,22)
    plt.ylim(-22,22)
    for j in range(n):
        plt.plot([cos(x)*(sum([h[i,j]*cos(f[i,j]*x+t) for i in range(10)])+15) for x in X],
                 [sin(x)*(sum([h[i,j]*sin(f[i,j]*x+t) for i in range(10)])+15) for x in X],
                 color=colors[j],
                 lw=2,
                 alpha = 0.85)

    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/24042020/plot{k}.PNG',facecolor='black')