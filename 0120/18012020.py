import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
def lines(n):
    p = plt.figure(figsize=(14,14),facecolor='black')
    p = plt.axis('off')
    f = lambda x: x**n
    for x in range(30):
        plt.plot([0,x],[f(x),f(x)],color=plt.cm.viridis(x/(30)),lw=3)
        plt.plot([x,x],[0,f(x)],color=plt.cm.plasma(x/(30)),lw=3)
for n,i in zip(list(np.linspace(0,5,250)) + list(np.linspace(5,0,250)),range(500)):
    lines(n)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/18012020/plor{i}.png',facecolor='black')