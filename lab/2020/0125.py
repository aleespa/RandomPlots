from math import cos, sin

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
for z in range(200):
    rango =range(z*10,(z+1)*10)
    plt.plot([(z)*cos(z)*sin(z*2) for z in rango],
             [z*sin(z)*sin(z) for z in rango],
             lw=4,
             alpha=0.7,
             color = plt.cm.RdPu(np.random.uniform(0,1)))
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/25012020.png',facecolor='black')