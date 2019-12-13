import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D

X, Y =  [t*cos(t) for t in np.linspace(0,100*pi,10000)],[sin(t) for t in np.linspace(0,100*pi,10000)]
colors  = ['#eebdff'	,'#d59dee','#a154c3','#632b7c','#5e0ba5']

p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-pi*100,pi*100)
p = plt.ylim(-1,1)
for i in range(180):
    plt.plot(X[i*55:(i+1)*55],Y[i*55:(i+1)*55],alpha=0.8,color= colors[i%5])
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/24112019/plor{i}.png',facecolor='black')