
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

fig, ax = plt.subplots(figsize=(14,14),facecolor='black',dpi=400)

p = plt.axis('off')
p = plt.xlim(0,2)
p = plt.ylim(0,2)
def circles(x,y,r,l):
    c = plt.cm.cool(np.random.uniform(0,1))
    for R in np.linspace(0,1,20):
        plt.plot([r*R*cos(t)+x for t in np.linspace(0,2*pi,6)],
                 [r*R*sin(t)+y for t in np.linspace(0,2*pi,6)],zorder=l+1,color=c,lw=2)
        ax.add_artist(plt.Circle((x,y),radius=r,color="black",zorder=l))
for i in range(100):
    circles(np.random.uniform(-0.2,2.2),np.random.uniform(0,2),0.5,i)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/09052020.PNG',facecolor='black')