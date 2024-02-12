
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh


n = 200000
X, Y = np.random.uniform(-1,1,n),np.random.uniform(-1,1,n)
X1,Y1 = [X[i] for i in range(n) if sqrt(X[i]**2 + Y[i]**2)<=1],[Y[i] for i in range(n) if sqrt(X[i]**2 + Y[i]**2)<1]
X2,Y2 = [X[i] for i in range(n) if sqrt(X[i]**2 + Y[i]**2)>1],[Y[i] for i in range(n) if sqrt(X[i]**2 + Y[i]**2)>=1]
print(4*len(X1)/(n),pi)

for i in range(1,666):
    X1,Y1 = [X[i] for i in range(0,333*i) if sqrt(X[i]**2 + Y[i]**2)<=1],[Y[i] for i in range(0,333*i) if sqrt(X[i]**2 + Y[i]**2)<=1]
    X2,Y2 = [X[i] for i in range(0,333*i) if sqrt(X[i]**2 + Y[i]**2)>1],[Y[i] for i in range(0,333*i) if sqrt(X[i]**2 + Y[i]**2)>1]
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    p = plt.scatter(X1,Y1,color='red',alpha=0.8,s=7)
    p = plt.scatter(X2,Y2,color='#aaaaaa',alpha=0.8,s=7)
    # p = plt.text(s=f'$\pi\\approx${4*len(X1)/(333*i):.5f}...',x=-0.35,y=0,size=25.5,color='white')
    p = plt.text(s=f'$\pi\\approx${4*len(X1)/(333*i):.6f}...',x=-0.29,y=0,size=29,color='white')
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/14032020/plot{i}.PNG',facecolor='black')