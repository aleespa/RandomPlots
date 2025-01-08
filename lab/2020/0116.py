from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

colors = ['#f28db3','#69edde',	'#ffddff'	,'#ffffff'	,'#ED5F94']*20

def gal(u):
    p = plt.figure(figsize=(14,14),facecolor='black')
    p = plt.axis('off')
    for i in np.linspace(0,5,120):
        plt.scatter([i*sin(x+u*i) for x in np.linspace(0,6*pi,100)],
                    [i*cos(x+i) for x in np.linspace(0,6*pi,100)],s=10,alpha=0.8,color=colors)

for u, i in zip([x**2 for x in np.linspace(0,5,600)],range(600)):
    gal(u)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/16012020/plor{i}.png',facecolor='black')