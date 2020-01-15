import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for x,y,z in [(5*cos(t),5*sin(t),t) for t in np.linspace(0,2*pi)]:
    plt.scatter(np.random.normal(x,1.5,400),np.random.normal(y,1.5,400),s=np.random.uniform(5,100,400),alpha=0.6,
                color=plt.cm.winter(z/(2*pi)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/11012020.png',facecolor='black')