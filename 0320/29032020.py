import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh
from cmath import exp as cexp, log as clog

n = 3000
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
plt.xlim(0,3000)
plt.ylim(0,3000)
plt.scatter([x*cos(x) for x in range(n)],
            [x*sin(x) for x in range(n)],s=25,lw=1,alpha=0.9,color='#ff1166')
plt.scatter([x*cos(x)+3000 for x in range(n)],
            [x*sin(x) +3000 for x in range(n)],s=25,lw=1,alpha=0.9,color='#f16f4d')
plt.scatter([x*cos(x) for x in range(n)],
            [x*sin(x) +3000 for x in range(n)],s=25,lw=1,alpha=0.9,color='#f4df16')
plt.scatter([x*cos(x) +3000 for x in range(n)],
            [x*sin(x)  for x in range(n)],s=25,lw=1,alpha=0.9,color='#00c130')

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/29032020.PNG',facecolor='black')