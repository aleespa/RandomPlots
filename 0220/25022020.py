import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh

for l,i in zip(np.linspace(0,3,690),range(690)):
    p = plt.figure(figsize=(14,14),facecolor='black',dpi=100)
    p = plt.axis('off')
    p = plt.xlim(-40,40)
    p = plt.ylim(-10,470)
    for t in np.linspace(0,0.5,40):
        plt.plot([x*cos(x*l) for x in np.linspace(0,40,400)],
                 [x*sin(x*t)+10*x for x in np.linspace(0,40,400)],
                 alpha=0.6,
                 lw=np.random.uniform(0.5,3),
                 color=plt.cm.gray(t*2))
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/25022020/plot{i}.png',facecolor='black')