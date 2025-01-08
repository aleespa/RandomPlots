from math import cos, sin

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
plt.scatter([(z)*cos(z)*sin(z) for z in range(7000)],
            [z*sin(z)*sin(z) for z in range(7000)],
            s=4,
            color=[plt.cm.Accent(np.random.uniform(0,1)) for z in range(7000)])
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23012020_1.png',facecolor='black')

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
plt.scatter([(z)*cos(z)*sin(z*10) for z in range(7000)],
            [z*sin(z)*sin(z) for z in range(7000)],
            s=4,
            color=[plt.cm.Accent(np.random.uniform(0,1)) for z in range(7000)])
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23012020_2.png',facecolor='black')

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
plt.scatter([(z)*cos(z)*sin(z*10) for z in range(7000)],
            [z*sin(z)*sin(z*.4) for z in range(7000)],
            s=4,
            color=[plt.cm.Accent(np.random.uniform(0,1)) for z in range(7000)])
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/23012020_3.png',facecolor='black')