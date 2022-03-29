import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
plt.xlim(-0.15,1.15)
# plt.ylim(0,1)
s = 0.25
k=60
colors= ['#ecdda3', '#f6fdb4', '#c9f3a6','#87e4a3' ,'#a8a495']
for z in np.linspace(0,250,40):
    h = np.random.choice([-1,0,1],k)
    f = np.random.uniform(2,30,k)
    plt.plot(np.linspace(0,1,500),
             [z+1/sqrt(s*2*pi)*exp(-((x-0.5)/s)**2)*sum([h[i]*cos(f[i]*x*2*pi) for i in range(k)]) for x in np.linspace(0,1,500)],
             color=np.random.choice(colors),lw=2)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/22042020.PNG',facecolor='black')