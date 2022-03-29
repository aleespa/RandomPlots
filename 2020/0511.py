import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

def barra(x,y,c):
    plt.fill_between([x,x+1],[y,y],color=c)

n = 35
X = np.random.choice(range(1,n+1),n,replace=False)
l=0
for i in range(n-1):
    for j in range(0, n-i-1):
        if X[j] > X[j+1] :
            X[j], X[j+1] = X[j+1], X[j]
        fig, ax = plt.subplots(figsize=(13,13),facecolor='black',dpi=100)
        p = plt.axis('off')
        for k,x in enumerate(X):
            barra(k,x,c=plt.cm.rainbow(1-x/n))
        p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/11052020/plot{l}.PNG',facecolor='black')
        l +=1