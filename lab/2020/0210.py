
from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in np.linspace(0,1,100):
    plt.plot([z*cos(x)*np.random.uniform(0.95,1.05) for x in np.linspace(0,2*pi,1000)],
             [z*sin(x)*np.random.uniform(0.95,1.05) for x in np.linspace(0,2*pi,1000)],alpha=0.8,
             color = plt.cm.Blues(np.random.uniform(0,1)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/10022020.png',facecolor='black')