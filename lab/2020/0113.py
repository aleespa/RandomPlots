import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D

def helix(R,r,n):
    fig = plt.figure(figsize=(13,13),facecolor='black')
    ax = fig.gca(projection='3d',facecolor='black')
    ax.set_zlim(-5,5)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    p = plt.axis('off')
    t = np.linspace(0, 2*pi, 2000)
    x,y,z=(R+r*np.cos(n*t))*np.cos(t), (R+r*np.cos(n*t))*np.sin(t), r*np.sin(n*t)
    ax.plot(x, y, z,lw=6,alpha=1,color=plt.cm.hsv(n/70))
for l,i in zip(np.linspace(0,70,390),range(390)):
    helix(5,2,l)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/13012020/plor{i}.png',facecolor='black')