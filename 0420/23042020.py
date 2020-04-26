import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

p = plt.figure(figsize=(14,14),facecolor='black',dpi=500)
p = plt.axis('off')
plt.xlim(-290,290)
plt.ylim(-290,290)
colors= ['#0ac7f0', '#00cc89', '#fdea2d','white' ,'#b7c3c1']
for z in np.linspace(0,250,35):
    h = np.random.choice([-1,0,1],10)
    f = np.random.binomial(10,0.5,10)
    X = np.linspace(0,4*pi,500)
    plt.plot([x*cos(x)*(sum([h[i]*cos(f[i]*x) for i in range(10)])+15) for x in X],
             [x*sin(x)*(sum([h[i]*cos(f[i]*x) for i in range(10)])+15) for x in X],
             color=np.random.choice(colors))

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23042020.PNG',facecolor='black')