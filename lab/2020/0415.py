import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
p = plt.axis('off')
for u in np.linspace(0, 1, 30):
    plt.plot([0, 0, 1 - u, u, 0], [u, 0, 0, u, u], lw=4, color=plt.cm.PuRd(1 - u))
    plt.plot([0, 0, u, u, 0], [u, 0, 0, u, 1 - u], lw=4, color=plt.cm.PuRd(1 - u))

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/15042020.PNG', facecolor='black'
)
