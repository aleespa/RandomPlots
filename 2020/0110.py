import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for r in range(65):
    X = [0]
    Y = [0]
    for _ in range(2000):
        X.append(X[-1] + np.random.normal(0,2))
        Y.append(Y[-1] + np.random.normal(0,2))
    plt.plot([X[i]/(X[i]**2 + Y[i]**2)**0.5 for i in range(1,2000)],
             [Y[i]/(X[i]**2 + Y[i]**2)**0.5 for i in range(1,2000)],alpha=0.65,lw=1.2,color=plt.cm.hsv(r/58))
p=plt.plot([cos(x) for x in np.linspace(0,2*pi,100)],
           [sin(x) for x in np.linspace(0,2*pi,100)],alpha=1,color='white',zorder=160)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10012020.png',facecolor='black')