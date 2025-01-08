from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-400,400)
p = plt.ylim(-400,400)
def cubo(a,b,t,color):
    p = plt.plot([a*cos(t)-a*sin(t),b*cos(t)-a*sin(t),b*cos(t)-b*sin(t),a*cos(t)-b*sin(t),a*cos(t)-a*sin(t)],
                 [a*sin(t)+a*cos(t),b*sin(t)+a*cos(t),b*sin(t)+b*cos(t),a*sin(t)+b*cos(t),a*sin(t)+a*cos(t)],
                 color=plt.cm.rainbow(a/(400*pi**2)),alpha=1,lw=1.2)
for t in np.linspace(0,20*pi,180):
    cubo(t**2,t*2,t,'red')
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/05122019.png',facecolor='black')