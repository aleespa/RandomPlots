import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

colors = ['#79c562','#4ea150','#edf69f','#ecee43']
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for x in np.linspace(-1,1,30):
    for y in np.linspace(-1,1,30):
        plt.plot([x,y+x],[x,y-x],alpha = 0.75,lw=1,color = np.random.choice(colors))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/26122019.png',facecolor='black')