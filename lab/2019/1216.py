import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for y in range(4,80):
    Z = np.random.uniform(0,1,6)
    plt.fill_between([Z[0],Z[1]],
                     [Z[2],Z[3]],
                     [Z[4],Z[5]],alpha=0.5)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/16122019_1.png',facecolor='black')
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for y in range(4,100):
    Z = np.random.normal(0,1,6)
    plt.fill_between([Z[0],Z[1]],
                     [Z[2],Z[3]],
                     [Z[4],Z[5]],alpha=0.5)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/16122019_2.png',facecolor='black')
p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
for y in range(4,100):
    Z = np.random.gamma(1,3,6)
    plt.fill_between([Z[0],Z[1]],
                     [Z[2],Z[3]],
                     [Z[4],Z[5]],alpha=0.5)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/16122019_3.png',facecolor='black')