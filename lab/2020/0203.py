from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

lws = np.random.uniform(1,4.5,100)
colors = np.random.beta(0.5,0.5,100)
def hexline(t):
    p = plt.figure(figsize=(14,14),facecolor='black',dpi=100)
    p = plt.axis('off')
    for k,i in zip(np.linspace(0.5,22,100),range(100)):
        plt.plot([k*cos(z)+ k*cos(4*t)*0.5*(2*pi-t)**0.3 for z in np.linspace(0,2*pi,7)],
                 [k*sin(z)+k*sin(4*t)*0.5*(2*pi-t)**0.3 for z in np.linspace(0,2*pi,7)],
                 alpha=0.7,
                 lw=lws[i],
                 color=plt.cm.coolwarm_r(colors[i]))

for i, t in zip(range(300),np.linspace(0,2*pi,300)):
    hexline(t)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/03022020/plor{i}.png',facecolor='black')