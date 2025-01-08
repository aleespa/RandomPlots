from math import pi, exp, sqrt

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.ylim(-1,1)
for m in np.linspace(-10,10,25):
    plt.fill_between(np.linspace(-10,10,200),[m*0.2/(sqrt(2*pi))*exp(-(z-m)**2/2) for z in np.linspace(-10,10,200)],alpha=0.5)
    plt.fill_between(np.linspace(-10,10,200),[-m*0.2/(sqrt(2*pi))*exp(-(z-m)**2/2) for z in np.linspace(-10,10,200)],alpha=0.5)
    plt.scatter(np.linspace(-10,10,100),[-m*0.2/(sqrt(2*pi))*exp(-(z-m)**2/2) for z in np.linspace(-10,10,100)],alpha=0.5,s=1)
    plt.scatter(np.linspace(-10,10,100),[m*0.2/(sqrt(2*pi))*exp(-(z-m)**2/2) for z in np.linspace(-10,10,100)],alpha=0.5,s=1)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/04022020.png',facecolor='black')