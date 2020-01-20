import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for i in range(50):
    plt.scatter(np.random.normal(0,1,100),
                np.random.exponential(1,100),s=np.random.normal(45,70,100),
                alpha=0.7,
                color=plt.cm.YlOrRd(i/(50)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/17012020.png',facecolor='black')