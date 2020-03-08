import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh
from cmath import exp as cexp, log as clog

f = lambda x: x/110+ cos(x*15.751)
N = 12000
p = plt.figure(figsize=(12,12),facecolor='black',dpi=100)
p = plt.axis('off')
# plt.xlim(-2.5,1.5)
# plt.ylim(-1,3)
S = np.cumsum([cexp(f(n)*complex(0,2*pi)) for n in range(1,N)])
for z in range(100):
    plt.plot([S[i].real for i in range(N-1)][90*z:90*(z+1)+1],[S[i].imag for i in range(N-1)][90*z:90*(z+1)+1],lw=1,
             color=plt.cm.hsv(z/100+0.2),alpha=0.8)
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/02032020.PNG',facecolor='black')