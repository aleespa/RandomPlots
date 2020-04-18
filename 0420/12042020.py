import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog

f1 = lambda x:d*sqrt(sqrt(1+4*(x/d)**2)-(1+(x/d)**2))
f2 = lambda x:-f1(x)
fp = lambda x:x*(-1 + 2/sqrt(1 +(4*x**2)/(d**2)))/(d*sqrt(-1 - (x**2)/(d**2) + sqrt(1 + (4*x**2)/(d**2))))

d=1
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.ylim(-2,2)
p = plt.xlim(-2,2)
for i,a in enumerate(np.linspace(-d*sqrt(2)+0.001,0.0001,70)[::-1]):
    p = plt.plot(np.linspace(-4,4,10),
                 [f1(a) + fp(a)*(x-a) for x in np.linspace(-4,4,10)],
                 color=plt.cm.winter(1-(a+d*sqrt(2))/(d*sqrt(2))))
    p = plt.plot(np.linspace(-4,4,10),
                 [f2(a) + -fp(a)*(x-a) for x in np.linspace(-4,4,10)],
                 color=plt.cm.winter(1-(a+d*sqrt(2))/(d*sqrt(2))))

for i,a in enumerate(np.linspace(0.0001,d*sqrt(2)-0.001,70)[::-1]):
    p = plt.plot(np.linspace(-4,4,10),
                 [f1(a) + fp(a)*(x-a) for x in np.linspace(-4,4,10)],
                 color=plt.cm.winter((a)/(d*sqrt(2))))
    p = plt.plot(np.linspace(-4,4,10),
                 [f2(a) + -fp(a)*(x-a) for x in np.linspace(-4,4,10)],
                 color=plt.cm.winter((a)/(d*sqrt(2))))

plt.fill_between(np.linspace(0.0001,d*sqrt(2)-0.001),
                 [f1(x) for x in np.linspace(0.0001,d*sqrt(2)-0.001)],color='black',zorder=100)
plt.fill_between(np.linspace(0.0001,d*sqrt(2)-0.001),
                 [f2(x) for x in np.linspace(0.0001,d*sqrt(2)-0.001)],color='black',zorder=100)

plt.fill_between(np.linspace(-d*sqrt(2)+0.001,0.0001),
                 [f1(x) for x in np.linspace(-d*sqrt(2)+0.001,0.0001)],color='black',zorder=100)
plt.fill_between(np.linspace(-d*sqrt(2)+0.001,0.0001),
                 [f2(x) for x in np.linspace(-d*sqrt(2)+0.001,0.0001)],color='black',zorder=100)
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/12042020.PNG',facecolor='black')