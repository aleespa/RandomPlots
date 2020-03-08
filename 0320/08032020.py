import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh
for k,u in zip(range(900),np.linspace(2,6,900)):
    p = plt.figure(figsize=(13,13),facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-200,200)
    p = plt.ylim(-200,200)
    X,Y = [cos(x)/(sin(x*u)+2)*x for x in np.linspace(0,60*pi,10000)],[sin(x)/(cos(x*u)+2)*x for x in np.linspace(0,60*pi,10000)]
    for j in range(99):
        plt.scatter(X[j*100:(j+1)*100+1],Y[j*100:(j+1)*100+1],s=7,alpha=0.9,
                    color=[plt.cm.spring(i)*(j%8 in [0,1])+
                           plt.cm.summer(i)*(j%8 in [2,3])+
                           plt.cm.autumn(i)*(j%8 in [4,5])+
                           plt.cm.winter(i)*(j%8 in [6,7]) for i in np.linspace(0,1,101)])
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/08032020/plot{k}.PNG',facecolor='black')