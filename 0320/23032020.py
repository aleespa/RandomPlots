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
            c = np.random.uniform(0,1)
            plt.plot([x[0],x[0]+cos(t[k])],[x[1],x[1]+sin(t[k])],lw=2,alpha=0.8,color=plt.cm.viridis(c))
            plt.scatter([x[0],x[0]+cos(t[k])],[x[1],x[1]+sin(t[k])],s=2,color=plt.cm.viridis(c))
            Z.append([x[0]+cos(t[k]),x[1]+sin(t[k])])
    return(np.array(Z))

p = plt.figure(figsize=(14,14),facecolor='black',dpi=500)
p = plt.axis('off')
p = plt.xlim(-15,15)
p = plt.ylim(-15,15)

plt.scatter([0],[0],s=4,color='red',zorder=10000)
n = 1
p = plt.text(s=f'$n=$ {n}',x=-3,y=-9,size=25,color='red', bbox={'facecolor': 'black', 'alpha': 1, 'pad': 2})
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23032020/plot{0}.PNG',facecolor='black')
# p = plt.fill([-3.1,-3.1,3,3],[-9.1,-8.3,-8.3,-9.1],color='white',zorder=2)

X = branch(np.array([[0,0]]))
n +=len(X)
p = plt.text(s=f'$n=$ {n}',x=-3,y=-9,size=25,color='red', bbox={'facecolor': 'black', 'alpha': 1, 'pad': 2})
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23032020/plot{1}.PNG',facecolor='black')
# p = plt.fill([-3.1,-3.1,3,3],[-9.1,-8.3,-8.3,-9.1],color='white',zorder=2)


for i in range(2,21):
    X = branch(X)
    n +=len(X)
    p = plt.text(s=f'$n=$ {n}',x=-3,y=-9,size=25,color='red', bbox={'facecolor': 'black', 'alpha': 1, 'pad': 2})
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23032020/plot{i}.PNG',facecolor='black')
    # p = plt.fill([-3.1,-3.1,3,3],[-9.1,-8.3,-8.3,-9.1],color='white',zorder=2)