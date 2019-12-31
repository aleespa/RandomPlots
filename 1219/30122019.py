import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt


m = 0
for z in np.linspace(0,10,360):
    p = plt.figure(figsize=(13,13),facecolor='black')
    p = plt.axis('off')
    colors = ['#ff1496','#c3a6a8','#ff2387','#ff3278','#fbabb0']*400
    plt.scatter([i*cos(i) for i in range(2000)],[sin(i)*i**z for i in range(2000)],s=30,color = colors)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/30122019/plot{m}.png',facecolor='black')
    m+=1