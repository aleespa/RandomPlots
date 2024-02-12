import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
ran1 = np.random.beta(1.5,1,1000)*40*pi
p = plt.figure(figsize=(13,13),facecolor='black')
p = plt.axis('off')
line1,line2 =[(t)*sin(t)*cos(t) for t in np.linspace(0,80*pi,10000)], [(t)*cos(t)*t**2 for t in np.linspace(0,80*pi,10000)]
plt.plot(line1[:27*360],line2[:27*360],color='white')
for i in range(360):
    plt.plot(line1[27*i:27*(i+1)+1],line2[27*i:27*(i+1)+1],lw=3,alpha=0.9,color = plt.cm.plasma(np.random.uniform(0,1)))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/22122019/plot{i}.png',facecolor='black')
    print(i/360)