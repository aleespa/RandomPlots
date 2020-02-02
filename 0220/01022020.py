import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
# p = plt.xlim(0,6000)
# p = plt.ylim(0,6000)
for i in range(0,11):
    plt.scatter([z*cos(z) for z in range(10000) if z%15 ==i],
                [z*sin(z) for z in range(10000) if z%15 ==i],
                alpha=0.8,
                s=8,
                color=plt.cm.summer(i/10))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/01022020.png',facecolor='black')