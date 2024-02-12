from __future__ import division
import matplotlib.pylab as plt
import numpy as np
from math import sin,cos,pi,tan,sqrt
T = np.random.uniform(0,2*pi,580)
colors = ['#ff817a','#ff8d87','#ff9a94','#ffa6a1','#ffb3af','#ffc0bc','#ffccc9','#ffd9d7','#ffe5e4','#fff2f1','#ffffff']
colorslist = np.random.choice(colors,580)
def plots(s,name):
    p = plt.figure(figsize=(15,15),facecolor=(0, 0, 0),dpi=200)
    p = plt.axis('off')
    p = plt.xlim(-10,10)
    p = plt.ylim(-10,10)
    for m in range(1,10):
        for k in range(m*6+1):
            p = plt.plot([m*cos(T[k*m]+r) for r in np.linspace(0,0.5,50)],[m*sin(T[k*m]+r) for r in np.linspace(0,s,50)],
                     color=colorslist[k*m],alpha=0.8,lw=2)
            p = plt.scatter([m*cos(T[k*m])],[m*sin(T[k*m])],s=22,zorder=10,color='#00b0b0',alpha=0.8)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10112019/plot{name}.PNG',facecolor='black')
n =0
for i in np.linspace(0,2,180):
    plots(i,n)
    n+=1