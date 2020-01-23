import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

colors = [plt.cm.Spectral(np.random.uniform(0,1)) for j in range(60)]
def loc(z):
    p = plt.figure(figsize=(10,10),facecolor='black')
    p = plt.axis('off')
    p=plt.xlim(-1.1,1.1)
    p=plt.ylim(-1.1,1.1)
    for i,color in zip(np.linspace(0,1,60),colors):
        X,Y= [i*cos(y)*cos(z*y) for y in np.linspace(0,2*pi,1000)], [i*cos(y)*sin(z*y) for y in np.linspace(0,2*pi,1000)]
        plt.plot(X,Y,alpha=0.85,lw=2.2,color =color)
for z,i in zip(np.linspace(0,8,300),range(300)):
    loc(z)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/22012020/plor{i}.png',facecolor='black')