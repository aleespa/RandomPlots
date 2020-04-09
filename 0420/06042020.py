import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog
from mpl_toolkits.mplot3d import Axes3D

colors= ['#19d9b4','#92d275','#cccccc','#bbbbbb','#aaaaaa']
C = np.random.choice(colors,50)

for j,u in enumerate(list(np.linspace(0.00001,0.65,90)) + list(np.linspace(0.65,0.00001,90))[1:]):
    p = plt.figure(figsize=(13,13),facecolor='black',dpi=100)
    p = plt.axis('off')
    for i,z in enumerate(np.linspace(0,2*pi,50)):
        plt.plot([cos(x*z) +z/u for x in np.linspace(0,2*pi,80)],lw=3.5,color=C[i])
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/06042020/plot{j}.PNG',facecolor='black')