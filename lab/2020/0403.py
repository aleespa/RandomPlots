import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for n in range(1, 90):
    plt.scatter(
        np.linspace(0, 1, n),
        [n for i in range(n)],
        s=10,
        color=plt.cm.summer(1 - n / 100),
    )
    plt.scatter(
        np.linspace(0, 1, n),
        [-n for i in range(n)],
        s=10,
        color=plt.cm.spring(1 - n / 100),
    )

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/03042020.PNG', facecolor='black'
)
