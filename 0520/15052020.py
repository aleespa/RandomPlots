import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2,e
from cmath import exp as cexp, log as clog
from mpl_toolkits.mplot3d import Axes3D

for i,t in enumerate(np.linspace(0,2*pi,360)):
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d',facecolor='black')
    p = plt.axis('off')
    x = np.linspace(-2*pi,2*pi,1000)
    y = np.linspace(-2*pi,2*pi,1000)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.cos(xx*yy +6*t+pi)
    ax.view_init(t*180/(pi),t*180/(4*pi))
    ax.set_zlim(-3,3)
    ax.plot_surface(xx, yy, z, antialiased=True, shade=True,cmap=cm.winter,alpha=0.8)
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/15052020/plot{i}.PNG',facecolor='black')