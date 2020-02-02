import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for k in np.linspace(0.5,22,100):
    plt.plot([k*cos(z)+cos(k) for z in np.linspace(0,2*pi,1500)],
             [k*sin(z)+sin(k) for z in np.linspace(0,2*pi,1500)],
             alpha=0.8,
             lw=np.random.uniform(0.8,3),
             color=plt.cm.rainbow(((k+25)/50)))

plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/30012020.png',facecolor='black')