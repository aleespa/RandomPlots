from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-1.4,1.4)
colors=['#c0d8c6','#84d290','#80d091','#427c5f','#46ab9b']*100
for z in range(2,100):
    plt.plot([cos(z+y) for y in np.linspace(0,2*pi,400)],
             [y*z for y in np.linspace(0,2*pi,400)],
             lw=2.5,color=colors[z],alpha=0.8)
    plt.plot([cos(z+0.2*y) for y in np.linspace(0,2*pi,400)],
             [-y*z for y in np.linspace(0,2*pi,400)],
             lw=2.5,color=colors[z],alpha=0.8)
    plt.plot([cos(z+0.1*y) for y in np.linspace(0,2*pi,400)],
             [y*z+400 for y in np.linspace(0,2*pi,400)],
             lw=2.5,color=colors[z],alpha=0.9)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/20012020.png',facecolor='black')