from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for p in np.linspace(0,5,60):
    plt.scatter(np.linspace(0,1,900),[p*cos(x+p**2)+sin(x) for x in np.linspace(-2*pi,2*pi,900)],
                s=1,alpha=0.8,color =plt.cm.Spectral(np.random.uniform(0,1)-0.1))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/04012020.png',facecolor='black')