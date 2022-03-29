import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in np.linspace(-2*pi,2*pi,35):
    for w in np.linspace(-2*pi,2*pi,35):
        plt.plot([0.5*sin(z+w)*cos(x) +z for x in np.linspace(0,2*pi,300)],
                 [0.5*sin(z+w)*sin(x) + w for x in np.linspace(0,2*pi,300)],
                 alpha=0.9,color=np.random.choice(['#b8ffb9','#93eac4','#75dbd8','#6dc5df','#7eaee3']),
                 lw=2.2)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/13022020.png',facecolor='black')