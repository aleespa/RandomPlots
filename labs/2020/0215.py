import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D

theta = np.linspace(0, 2 * np.pi, 200)
w = np.linspace(-0.25, 0.25, 20)
w, theta = np.meshgrid(w, theta)
phi = 0.5 * theta
r = 1 + w * np.cos(phi)

x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))

tri = Triangulation(np.ravel(w), np.ravel(theta))
colors = [plt.cm.winter(np.random.uniform(0,1)) for i in range(4000)]

i=0
for t in np.linspace(0,360,300):
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca(projection='3d',facecolor='black')
    ax.scatter(x, y, z,s=32,color=colors)
    p = plt.axis('off')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.4, 0.4)
    ax.view_init(20,t)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/15022020/plot{i}.png',facecolor='black')
    i+=1