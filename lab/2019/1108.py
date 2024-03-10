import matplotlib.pylab as plt
import numpy as np
from math import sqrt , cos , sin , log , exp, pi

def pol(n):
    U = np.linspace(0,2*pi,n)
    plt.figure(figsize=(15,15),dpi=400)
    plt.xlim(-1.05,1.05)
    plt.ylim(-1.05,1.05)
    plt.axis('off')
    for j in range(n):
        for i in range(n):
            plt.plot([cos(U[j]), cos(U[i])],
                     [sin(U[j]), sin(U[i])], lw=0.7)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/08112019/pole{n}.PNG',facecolor='black')
pol(40)