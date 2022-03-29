
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh
from cmath import exp as cexp, log as clog
f = lambda x: x/64 + x*x*x/50
N = 12000
S = np.cumsum([cexp(f(n)*complex(0,2*pi)) for n in range(1,N)])

p = plt.figure(figsize=(14,14),facecolor='black',dpi=100)
p = plt.axis('off')
for k in range(330):
    plt.xlim(-2.5,1.5)
    plt.ylim(-1,3)
    plt.plot([S[i].real for i in range(N-1)][k*5:(k+1)*5+1],[S[i].imag for i in range(N-1)][k*5:(k+1)*5+1],lw=1,color=plt.cm.spring(k/300),alpha=0.8)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/01032020/plot{k}.PNG',facecolor='black')