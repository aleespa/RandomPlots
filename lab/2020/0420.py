import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

for i in range(510):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    x,y= np.random.normal(0,1,i**2+20),np.random.normal(0,1,i**2+20)
    p =plt.hist2d(x, y,bins=100,cmap='afmhot')
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/20042020/plot{i}.PNG',facecolor='black')