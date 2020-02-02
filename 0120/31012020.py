
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt




def BM(dt):
    T = np.arange(0,1+dt,dt)
    n = len(T)
    B = np.ones(n)*0
    for i in range(n):
        xi = sqrt(2)*(np.random.randn())/((i+0.5)*pi)
        B = B + xi*np.array([sin((i+0.5)*pi*t) for t in T])
    return T, B

def IntS(f, dt):
    T = np.arange(0,1+dt,dt)
    n = len(T)
    y = 0
    Y = [0]
    for i in range(1,n):
        y += f(T[i])*(T[i]-T[i-1])
        Y.append(y)
    return T, Y

def IntE(f, dt):
    T, B = BM(dt)
    n = len(T)
    y = 0
    Y = [0]
    for i in range(1,n):
        y += f(T[i])*(B[i]-B[i-1])
        Y.append(y)
    return T, Y




p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
colors= ['#3b698d','#4b87b6','#4ca4df','#37c6f6','#32e1f4','#00ffff']
for _ in range(110):
    plt.plot(IntE(lambda z:exp(-1/(z)),1/300)[1],np.linspace(0,1,301),
             alpha=0.9,
             color= np.random.choice(colors),
             lw=np.random.uniform(0.5,2))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/31012020.png',facecolor='black')