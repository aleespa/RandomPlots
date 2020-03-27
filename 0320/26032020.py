import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh,tanh

def Mandelbrot(a):
    z = 0
    for i in range(200):
        if abs(z)>2:
            break
        else:
            z = z**2+a
    if abs(z)>2:
        return(abs(z))

p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
colors= ['#60beb3','#7fffd4','#a6f6ea','#5b7bd6','#e6f5f4']*2
for u in np.linspace(-1.5,0.6,65):
    for x in np.linspace(-1,1,65):
        plt.scatter([u],[x],s=Mandelbrot(complex(u,x)),alpha=0.9,color=np.random.choice(colors))
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/26032020',facecolor='black')