from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in np.linspace(-3*pi,3*pi,200):
    plt.plot([0,0,z*cos(z),z*cos(z),0],[0,z*cos(z),z*cos(z),0,0],lw=1.2)
    plt.plot([0,0,-z*cos(z),-z*cos(z),0],[0,z*cos(z),z*cos(z),0,0],lw=1.2)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23012020.png',facecolor='black')