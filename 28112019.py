import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def BSp(x,y,n, Nij):
    z = 0
    for i in range(n):
        for j in range(n):
            z += Nij[i][j]*(2/(pi**2*(i-0.5)*(j-0.5)))*sin((i-0.5)*pi*x)*sin((j-0.5)*pi*y)
    return z

def BS(dt):
    T = np.arange(0,1+dt,dt)
    Z = pd.DataFrame(columns=T[1:])
    n = len(T)
    Nij = np.random.rand(n,n)
    for x in T[1:]:
        Z[x] = [BSp(x,y,n, Nij) for y in T[1:]]
    return  Z

z = BS(0.01).values

for t in range(360):
    fig = plt.figure(figsize=(12,12),dpi=200)
    ax = fig.add_subplot(111, projection='3d',facecolor='black')
    ax.set_zlim(0.6,2.1)
    p = plt.axis('off')
    ax.view_init(0, t)
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    xx, yy = np.meshgrid(x, y, sparse=True)
    ax.plot_surface(xx, yy, z,linewidth=0, antialiased=True, shade=True,cmap=cm.inferno,alpha=0.5)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/28112019/plor{t}.png',facecolor='black')