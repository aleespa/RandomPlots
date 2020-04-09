import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm,chi2,binom,gamma
import seaborn as sns
from math import cos, sin,log,tan,pi,exp,sqrt,cosh,sinh,tanh,atan,atan2
from cmath import exp as cexp, log as clog


p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-300,300)
p = plt.ylim(-300,300)
def cubo(a,b,t,color):
    p = plt.plot([a*cos(t)-a*sin(t),b*cos(t)-a*sin(t),b*cos(t)-b*sin(t),a*cos(t)-b*sin(t),a*cos(t)-a*sin(t)],
                 [a*sin(t)+a*cos(t),b*sin(t)+a*cos(t),b*sin(t)+b*cos(t),a*sin(t)+b*cos(t),a*sin(t)+a*cos(t)],
                 color=plt.cm.twilight_shifted(a/(35*pi)-0.1),alpha=1,lw=3)
for t in np.linspace(0,35*pi,150):
    cubo(t,t*2,0.1*t,'red')

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/07042020.PNG',facecolor='black')