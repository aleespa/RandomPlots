import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
colors = ['#e1d18e','#ce4343','#dd653c','#6d45d6','#f4c0a9','#77bd98','#00c2c7']
n = 0
for p in np.linspace(0,4,60):
    plt.plot([cos(x) for x in np.linspace(0,4*pi,1000)],[cos(x*p)+sin(x*p) + sin(x) for x in np.linspace(0,4*pi,1000)],
             lw=2,color =np.random.choice(colors),alpha=0.8)
    n+=1
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/05012020.png',facecolor='black')