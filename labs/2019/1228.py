import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

colors = ['#32886f','#0b5a42','#72b7a3','#bbbbbb','#aaaaaa']*100
n = 0
m = 0
for pol in [4,5,7]:
    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')
    for z in np.linspace(0,2*pi,85):
        plt.plot([cos(x)+2*cos(z) for x in np.linspace(0,2*pi,pol)],
                 [sin(x)+2*sin(z) for x in np.linspace(0,2*pi,pol)],lw=6,alpha=0.75,color = colors[n])
        n+=1
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/28122019/plot{m}.png',facecolor='black')
    m += 1