import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog


def brownian_path(N):
    Δt_sqrt = sqrt(1 / N)
    Z = np.random.randn(N)
    Z[0] = 0
    B = np.cumsum(Δt_sqrt * Z)
    return B
n = 50
X, Y = [brownian_path(600) for i in range(n)],[brownian_path(600) for i in range(n)]
p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
p = plt.axis('off')
plt.xlim(-2,2)
plt.ylim(-2,2)
for k in range(600):
    p = plt.scatter([x[k] for x in X],[y[k] for y in Y],alpha=0.8,color=plt.cm.rainbow(k/600))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/16032020/plot{k}.PNG',facecolor='black')