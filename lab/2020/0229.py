import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh
from cmath import exp as cexp, log as clog
N = 7500
f = lambda n:n/11+n**2/21+n**3/31
S = np.cumsum([cexp(f(n)*complex(0,2*pi)) for n in range(1,N)])

p = plt.figure(figsize=(14,14),facecolor='black',dpi=100)
p = plt.axis('off')
for k in range(440):
    plt.ylim(-25,35)
    plt.xlim(-40,20)
    plt.plot([S[i].real for i in range(N-1)][k*20:(k+1)*20+1],[S[i].imag for i in range(N-1)][k*20:(k+1)*20+1],lw=1,color='w',alpha=0.8)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/29022020/plot{k}.PNG',facecolor='black')