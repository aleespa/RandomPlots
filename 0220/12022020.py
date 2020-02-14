import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-1,1)
p = plt.ylim(-1,1)
for z in np.linspace(0,2*pi,91):
    plt.fill_between([0,cos(z-pi/2)],[0,sin(z-pi/2)],alpha=0.3,color=plt.cm.RdYlBu(z/(2*pi)),lw=0.2)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/12022020.png',facecolor='black')