import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in range(200):
    rango =range(z*10,(z+1)*10)
    plt.scatter([z*sin(z)*cos(z)*cos(z) for z in rango],
                [cos(z)*sin(z)*z**2 for z in rango],
                lw=4.5,
                alpha=0.7,
                color = plt.cm.Wistia(np.random.uniform(0,1)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/27012020.png',facecolor='black')