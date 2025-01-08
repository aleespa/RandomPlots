from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14,14),facecolor='black',dpi=500)
p = plt.axis('off')
def drawCircle(x,y,r):
    plt.plot([r*cos(t)+x for t in np.linspace(0,2*pi,70)],
             [r*sin(t)+y for t in np.linspace(0,2*pi,70)],lw=1,color="white")
    if r>2:
        drawCircle(x + r/2, y, r/2)
        drawCircle(x , y + r/2, r/2)
        drawCircle(x , y - r/2, r/2)
        drawCircle(x - r/2, y, r/2)
drawCircle(0,0,50)

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/04052020_V2.PNG',facecolor='black')