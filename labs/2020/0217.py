import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
def BB(n):
    T = np.linspace(0,1,n)
    B = np.ones(n)*0
    for i in range(n):
        xi = sqrt(2)*np.random.randn()/((i+1)*pi)
        B = B + xi*np.array([sin((i+1)*pi*t) for t in T])
    return B
n = 1000
color1 = ['#96ceb4', '#ffeead','#ff6f69','#ffcc5c','#88d8b0']
color2 = ['#f7f4a3', '#7fccec','#6a81d9','#a479c9','#dfdfdf']

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for k in range(55):
    X = BB(n)
    p = plt.plot(np.linspace(0,1,n),[X[i]+(k)*i/n - k for i in range(n)],color =np.random.choice(color1),
                 lw=1,alpha=0.8)
    p = plt.plot(np.linspace(0,1,n),[-(X[i]+(k)*i/n - k) for i in range(n)],color =np.random.choice(color1),
                 lw=1,alpha=0.8)
    p = plt.plot(-np.linspace(0,1,n),[-(X[i]*6+(k)*i/n - k) for i in range(n)],color =np.random.choice(color2),
                 lw=0.5,alpha=1)
    p = plt.plot(-np.linspace(0,1,n),[(X[i]*6+(k)*i/n - k) for i in range(n)],color =np.random.choice(color2),
                 lw=0.5,alpha=1)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17022020.png',facecolor='black')