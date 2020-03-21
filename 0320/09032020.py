import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in np.linspace(0,1,50):
    plt.scatter([z*cos(x)*sin(x)*sin(3*x) for x in np.linspace(1,3*pi,500)],
                [z*cos(x)*sin(x)*cos(3*x) for x in np.linspace(1,3*pi,500)]
                ,alpha=0.9,color=plt.cm.gist_heat_r(z-0.1),s=z*8)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/09032020.png',facecolor='black')