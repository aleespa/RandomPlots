import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D

R = np.random.normal(10,1,10000)
def lemis1(t1,t2):
    fig = plt.figure(figsize=(13,13),facecolor='black')
    ax = fig.gca(projection='3d',facecolor='black')
    p = plt.axis('off')
    t = np.linspace(0, 2*pi, 10000)
    z = np.linspace(-2, 2, 10000)
    x,y,z=(np.cos(t+pi)), np.sin(t+pi)*np.cos(t), 2**R*np.sin(t*4)
    ax.plot(x, y, z,lw=0.8,alpha=1,color=plt.cm.hsv(30/100))
    ax.view_init(t1, t2)
n = 0
for t in np.linspace(0,180,180):
    lemis1(0,t)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/14012020/plor{n}.png',facecolor='black')
    n+=1
for t in np.linspace(0,360,360):
    lemis1(t,180)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/14012020/plor{n}.png',facecolor='black')
    n+=1