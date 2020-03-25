import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog

def branch(X):
    Z = []
    for x in X:
        K = np.random.randint(1,3)
        t = np.random.uniform(0,2*pi,K)
        for k in range(K):
            plt.plot([x[0],x[0]+cos(t[k])],[x[1],x[1]+sin(t[k])],lw=1,color='#fff4df',alpha=0.5)
            plt.scatter([x[0],x[0]+cos(t[k])],[x[1],x[1]+sin(t[k])],s=np.random.uniform(2,65),color=np.random.choice(['#ff69b4','#00bfff','#69ffb4']),zorder=1000000,alpha=0.8)
            Z.append([x[0]+cos(t[k]),x[1]+sin(t[k])])
    return(np.array(Z))


p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
# p = plt.xlim(-10,10)
# p = plt.ylim(-10,10)
X = branch(np.array([[0,0]]))
for _ in range(15):
    X = branch(X)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/24032020',facecolor='black')