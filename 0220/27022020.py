import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt,cosh,sinh

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-2,2)
p = plt.ylim(-2,2)
color1= ['#003087','#005eb8','#0072ce','#41b6e6','#00a9ce']
color2= ['#ffa500','#c83200','#ffb76d','#ffa264','#ff875f']
for z in range(1000):
    plt.plot(np.repeat(np.random.normal(0,0.8),20),
             np.linspace(-1,1,20),
             alpha=0.15,
             lw=np.random.uniform(0.5,5),
             color=np.random.choice(color1))
    plt.plot(np.linspace(-1,1,20),
             np.repeat(np.random.normal(0,0.8),20),
             alpha=0.15,
             lw=np.random.uniform(0.5,5),
             color=np.random.choice(color2))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/27022020.png',facecolor='black')