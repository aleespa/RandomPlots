
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

def rot(year):
    p = plt.figure(figsize=(13,13),facecolor='black')
    p = plt.axis('off')
    plt.scatter([0],[0],s=1500,color='yellow')
    plt.scatter([0],[0],s=500,color='orange')
    plt.scatter([0],[0],s=150,color='red')
    p = plt.xlim(-161,161)
    p = plt.ylim(-161,161)
    if year==2020:
        plt.text(-10,25, int(year), fontsize=28,color='red')
    else:
        plt.text(-10,25, int(year), fontsize=18,color='white')
    #Tierra****************
    plt.plot([149.6*cos(x) for x in np.linspace(0,2*pi,200)],[149.6*sin(x) for x in np.linspace(0,2*pi,200)],color='w',alpha=1,
             ls='--',lw=2)
    plt.scatter([149.6*cos(year*2*pi)],[149.6*sin(year*2*pi)],color='#4c89ff',s=280,zorder=10)
    #Luna
    plt.plot([10*cos(x) + 149.6*cos(year*2*pi) for x in np.linspace(0,2*pi,200)],
             [10*sin(x)+149.6*sin(year*2*pi)  for x in np.linspace(0,2*pi,200)],color='w',alpha=1,
             ls='--',lw=1)
    plt.scatter([10*cos(2*pi*year*13)+149.6*cos(year*2*pi)],
                [10*sin(2*pi*year*13)+149.6*sin(year*2*pi)],
                color='white',s=30,zorder=10)
    #Mercurio
    plt.plot([57.91*cos(x) for x in np.linspace(0,2*pi,200)],[57.91*sin(x) for x in np.linspace(0,2*pi,200)],color='w',alpha=0.8,
             ls='--',lw=1)
    plt.scatter([57.91*cos(year*2*pi*4+1.25)],[57.91*sin(year*2*pi*4+1.25)],color='grey',s=150,zorder=10)
    #Venus
    plt.plot([108.2*cos(x) for x in np.linspace(0,2*pi,200)],[108.2*sin(x) for x in np.linspace(0,2*pi,200)],color='w',alpha=0.8,
             ls='--',lw=1)
    plt.scatter([108.2*cos(year*2*pi*1.66+2.18)],[108.2*sin(year*2*pi*1.66+2.18)],color='#d8b750',s=170,zorder=10)
m =0
for y in np.linspace(0,2010,210):
    rot(y)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/31122019/plot{m}.png',facecolor='black')
    m+=1
for y in np.linspace(2010,2018,190):
    rot(y)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/31122019/plot{m}.png',facecolor='black')
    m+=1
for y in np.linspace(2018,2020,120):
    rot(y)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/31122019/plot{m}.png',facecolor='black')
    m+=1
for _ in range(60):
    rot(2020)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/31122019/plot{m}.png',facecolor='black')
    m+=1