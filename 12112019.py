import matplotlib.pylab as plt
import numpy as np
from math import cos, sin,log,tan,gamma,pi,exp
colors = ['#ff9b9b','#f8ff90','#a9ff8f','#22ba5a','#58c0e7']
p = plt.figure(figsize=(11,11),facecolor='black',dpi=400)
p = plt.axis('off')
p = plt.xlim(-260,260)
p = plt.ylim(-260,260)
def cubo(a,b,t,color):
    p = plt.plot([a*cos(t)-a*sin(t),b*cos(t)-a*sin(t),b*cos(t)-b*sin(t),a*cos(t)-b*sin(t),a*cos(t)-a*sin(t)],
                 [a*sin(t)+a*cos(t),b*sin(t)+a*cos(t),b*sin(t)+b*cos(t),a*sin(t)+b*cos(t),a*sin(t)+a*cos(t)],
                 color=np.random.choice(colors),alpha=0.8)
n=120
for t in np.linspace(0,20*pi,120)[::-1]:
    cubo(1*t,3*t,t,'red')
    p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/12112019/plot{n}.PNG',facecolor='black')
    n+=1