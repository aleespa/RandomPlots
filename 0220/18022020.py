import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D



i=0
for z in np.linspace(0,2,900):
    p = plt.figure(figsize=(14,14),facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-280,280)
    p = plt.ylim(-280,280)
    def cubo(a,b,t,color):
        p = plt.plot([a*cos(t)-a*sin(t),b*cos(t)-a*sin(t),b*cos(t)-b*sin(t),a*cos(t)-b*sin(t),a*cos(t)-a*sin(t)],
                     [a*sin(t)+a*cos(t),b*sin(t)+a*cos(t),b*sin(t)+b*cos(t),a*sin(t)+b*cos(t),a*sin(t)+a*cos(t)],
                     color=plt.cm.cool(a/(20*pi)-0.3),alpha=0.8,lw=3)
    for t in np.linspace(0,20*pi,80):
        cubo(t,3*t+1,t*z,'red')
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/18022020/plot{i}.png',facecolor='black')
    i+=1
