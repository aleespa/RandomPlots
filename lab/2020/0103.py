from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(12,12),facecolor='black',dpi=800)
p = plt.axis('off')
n=0
for p in np.linspace(0,5,60):
    plt.plot(np.linspace(0,1,800),[cos(x+p)+sin(x) for x in np.linspace(-3*pi,3*pi,800)],
             lw=6,alpha=0.8,color =plt.cm.rainbow(np.random.uniform(0,1)-0.1))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/03012020.png',facecolor='black')