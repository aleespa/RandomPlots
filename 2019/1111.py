import matplotlib.pylab as plt
import numpy as np
from math import sin,cos,pi,tan,sqrt

colors1 = [ '#ccabd8','#8474a1','#6ec6ca','#08979d','#055b5c']
colors2= ['#ff00b4','#00ffbc','#8ea5ff'	,'#ffffff','#c493ff']

p = plt.figure(figsize=(12, 12), facecolor='black',dpi=400)
p = plt.axis('off')
for x in range(30):
    for y in range(30):
        p1 = np.array([x,y])
        r1 = np.random.choice([-1,0,1])
        r2 = np.random.choice([-1,0,1])
        p = plt.plot([p1[0],p1[0]+r2],[p1[1],p1[1]+r1],color = np.random.choice(colors1))
        p = plt.plot([p1[0],p1[0]-r1],[p1[1],p1[1]-r2],color = np.random.choice(colors1))

p = plt.savefig('C:/Users/Alejandro/Pictures/RandomPlots/11112019_1.PNG',facecolor='black')

p = plt.figure(figsize=(12, 12), facecolor='black',dpi=400)
p = plt.axis('off')
for x in range(30):
    for y in range(30):
        p1 = np.array([x,y])
        for q in  np.random.choice([-1,0,1],2,p=[0.2,0.6,0.2]):
            p = plt.plot([p1[0],p1[0]],[p1[1],p1[1]+q],color = np.random.choice(colors2))
        for q in  np.random.choice([-1,0,1],2,p=[0.2,0.6,0.2]):
            p = plt.plot([p1[0],p1[0]+q],[p1[1],p1[1]],color = np.random.choice(colors2))
p = plt.savefig('C:/Users/Alejandro/Pictures/RandomPlots/11112019_2.PNG',facecolor='black')