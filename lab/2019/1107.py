import matplotlib.pylab as plt
import numpy as np
from math import sqrt, cos, sin, log, exp, pi

T = np.linspace(0, pi / 4, 200)

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.xlim(-10, 10)
p = plt.ylim(-10, 10)
for k in np.linspace(0, 10, 50):
    X, Y = np.array([(4 + k) * sqrt(cos(2 * t)) * cos(t + k / 2) for t in T]), np.array(
        [(4 + k) * sqrt(cos(2 * t)) * sin(t + k / 2) for t in T])
    p = plt.axis('off')
    p = plt.plot(X, Y, color=np.random.choice(
        ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']), lw=1)
    p = plt.plot(-X, Y, color=np.random.choice(
        ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']), lw=1)
    p = plt.plot(-X, -Y, color=np.random.choice(
        ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']), lw=1)
    p = plt.plot(X, -Y, color=np.random.choice(
        ['#23895d', '#7fd677', '#eadb87', '#eeda9e', '#b8b8b8']), lw=1)
X, Y = np.array([4 * sqrt(cos(2 * t)) * cos(t) for t in T]), np.array(
    [4 * sqrt(cos(2 * t)) * sin(t) for t in T])
p = plt.plot(X, Y, lw=7, color=(0, 0, 0))
p = plt.plot(X, -Y, lw=7, color=(0, 0, 0))
p = plt.plot(-X, -Y, lw=7, color=(0, 0, 0))
p = plt.plot(-X, Y, lw=7, color=(0, 0, 0))
p = plt.savefig('C:/Users/Alejandro/Pictures/RandomPlots/07112019.PNG', facecolor='black')
