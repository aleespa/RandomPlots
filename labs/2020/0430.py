import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

p = plt.figure(figsize=(15,15),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.ylim(0,1)
for i,z in enumerate(np.linspace(-4,4,150)):
    p = plt.plot(np.linspace(-4,4,2),[norm.cdf(z) + norm.pdf(z)*(x-z) for x in np.linspace(-4,4,2)],
                 color=plt.cm.RdPu((z+4)/(8)),lw=2)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/30042020.PNG',facecolor='black')