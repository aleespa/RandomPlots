from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
colors=['#f7f2cc','#fff77f','#f7ed53','#a8d5ca','#6a8d85']*30
for z in range(2,100):
    plt.plot([cos(y-z)*sin(y+z) for y in np.linspace(0,2*pi,400)],[cos(y+z) for y in np.linspace(0,2*pi,400)],
             lw=2.1,color=colors[z],alpha=1)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/19012020.png',facecolor='black')