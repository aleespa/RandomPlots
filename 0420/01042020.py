import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2

colors= ['#ffd58e','#ff9b5a','#f26249','#ec2f3b','#52b3b6']*(30*30)

for t,i in zip(np.linspace(0,2*pi,120),range(120)):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    k=0
    for z in range(1,30):
        for w in range(1,30):
            x,y = 15+10*cos(t),10*sin(t)+15
            plt.plot([z,z-cos(atan2(w-y,z-x))],[w,w-sin(atan2(w-y,z-x))],lw=2.5,
                     color=colors[k])
            k+=1
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/01042020/plot{i}.PNG',facecolor='black')