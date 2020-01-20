import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import cos, sin,log,tan,gamma,pi,exp,sqrt
from mpl_toolkits.mplot3d import Axes3D


r1,r2,r3 = np.random.uniform(2,8,20),np.random.uniform(0.8,1.3,20),np.random.uniform(0.5,1,20)
r4,r5 = np.random.uniform(0.5,1,20),np.random.uniform(8,20,20)

def rayos(u):
    p = plt.figure(figsize=(14,14),facecolor='black')
    p = plt.axis('off')
    plt.ylim(-2,2)
    for i in range(20):
        plt.plot(np.linspace(0,2*pi,1000),[r3[i]*cos((t)*r2[i]+r1[i]*u ) for t in np.linspace(0,2*pi,1000)],
                 alpha=r4[i],
                 lw = r5[i])
for u, i in zip(np.linspace(0,6*pi,500),range(500)):
    rayos(u)
    plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/15012020/plor{i}.png',facecolor='black')