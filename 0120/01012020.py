import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from math import cos, sin,log,tan,gamma,pi,exp,sqrt

def branch(n,m):
    plt.plot([n,n+1],[m,m+1],color='#db3a42',lw=2,alpha=0.4)
    plt.plot([n,n+1],[m,m-1],color='#db3a42',lw=2,alpha=0.4)

n = 50
p = plt.figure(figsize=(13,13),facecolor='black')
p = plt.axis('off')
p = plt.plot([x for x in np.linspace(0,n-2,150)],[sqrt(x) for x in np.linspace(0,n-2,150)],
             color='#32ffb9',lw=5,zorder=n**2,alpha=0.8)
p = plt.plot([x for x in np.linspace(0,n-2,150)],[-sqrt(x) for x in np.linspace(0,n-2,150)],
             color='#32ffb9',lw=5,zorder=n**2,alpha=0.8)
for j in range(n):
    if j%2 == 0:
        for i in [k for k in range(j) if k%2==0] + [-k for k in range(j) if k%2==0]:
            branch(j-2,i)
    else:
        for i in [k for k in range(j) if k%2==1] + [-k for k in range(j) if k%2==1]:
            branch(j-2,i)
m = 210
k=0
for _ in range(m):
    S = [0]
    for i in range(n-2):
        S.append(S[-1]+np.random.choice([-1,1]))
    plt.plot(range(n-1),S,color='white',alpha=0.35,lw=4)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/01012020/plot{k}.png',facecolor='black')
    k+=1
plt.text(0,n/2, "$\mathrm{}Var(S_t) \\approx t$", fontsize=22,color='white')
for _ in range(60):
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/01012020/plot{k}.png',facecolor='black')
    k+=1
