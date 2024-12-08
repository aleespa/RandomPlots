import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D


for n in range(4,8):
    n=6
    p = plt.figure(figsize=(12,12),facecolor='black',dpi=500)
    p = plt.axis('off')
    p = plt.xlim(-2,2)
    p = plt.ylim(-2,2)
    plt.plot([cos(x) for x in np.linspace(0,2*pi,n)],
             [sin(x) for x in np.linspace(0,2*pi,n)],lw=3,alpha=1)

    for y in np.linspace(0,2*pi,n):
        plt.plot([(cos(x))*0.5+cos(y) for x in np.linspace(0,2*pi,n)],
                 [(sin(x))*0.5 +sin(y) for x in np.linspace(0,2*pi,n)],lw=2,alpha=1)
        for z in np.linspace(0,2*pi,n):
            plt.plot([(cos(x))*0.25+cos(y)*0.5+cos(z) for x in np.linspace(0,2*pi,n)],
                     [(sin(x))*0.25 +sin(y)*0.5+sin(z) for x in np.linspace(0,2*pi,n)],lw=1.8,alpha=1)
            for w in np.linspace(0,2*pi,n):
                plt.plot([(cos(x))*0.125+cos(y)*0.5+cos(z)*.25+cos(w) for x in np.linspace(0,2*pi,n)],
                         [(sin(x))*0.125 +sin(y)*0.5+sin(z)*.25+sin(w) for x in np.linspace(0,2*pi,n)],lw=1,alpha=1)
                for u in np.linspace(0,2*pi,n):
                    plt.plot([(cos(x))*0.0625+cos(y)*0.5+cos(z)*.25+cos(w)*0.125+cos(u) for x in np.linspace(0,2*pi,n)],
                             [(sin(x))*0.0625 +sin(y)*0.5+sin(z)*.25+sin(w)*0.125+sin(u)  for x in np.linspace(0,2*pi,n)],lw=1,alpha=1)
                    if n < 6:
                        for r in np.linspace(0,2*pi,n):
                            plt.plot([(cos(x))*0.0625/2+cos(y)*0.5+cos(z)*.25+cos(w)*0.125+cos(u)*0.0625+cos(r) for x in np.linspace(0,2*pi,n)],
                             [(sin(x))*0.0625/2 +sin(y)*0.5+sin(z)*.25+sin(w)*0.125+sin(u)*0.0625+sin(r)  for x in np.linspace(0,2*pi,n)],lw=1,alpha=1)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17122019/plot{n}.png',facecolor='black')