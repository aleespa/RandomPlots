import matplotlib.pylab as plt
import numpy as np
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
colors =['#f12525','#fe6d57','#f34126','#f95312','#ffe7d3']
def circle(t,r,n,z):
    for x in np.linspace(-r,r,n):
        y = sqrt(r**2-x**2)
        p = plt.plot([x*cos(t)-y*sin(t),x*cos(t)+y*sin(t)],
                     [x*sin(t)+y*cos(t),x*sin(t)-y*cos(t)],
                     alpha=0.8,
                     zorder = z,
                     color= np.random.choice(colors))

p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-1,1)
p = plt.ylim(-1,1)
circle(0,1,60,1)
circle1=plt.Circle((0,0),.75,color='black',zorder=2)
p =plt.gcf().gca().add_artist(circle1)
circle(pi/2,0.75,55,3)
circle1=plt.Circle((0,0),.5,color='black',zorder=4)
p =plt.gcf().gca().add_artist(circle1)
circle(pi/4,0.5,40,5)
circle1=plt.Circle((0,0),.25,color='black',zorder=6)
p =plt.gcf().gca().add_artist(circle1)
circle(0,0.25,30,7)
circle1=plt.Circle((0,0),.125,color='black',zorder=8)
p =plt.gcf().gca().add_artist(circle1)
circle(pi/2,0.125,30,9)
circle1=plt.Circle((0,0),.0625,color='black',zorder=10)
p =plt.gcf().gca().add_artist(circle1)
circle(pi/4,0.0625,30,11)
circle1=plt.Circle((0,0),0.03125,color='black',zorder=12)
p =plt.gcf().gca().add_artist(circle1)
circle(0,0.03125,30,13)
p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/16112019.png',facecolor='black')