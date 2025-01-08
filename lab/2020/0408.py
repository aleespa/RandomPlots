import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for k in range(2, 30):
    plt.scatter(
        np.linspace(0, 1),
        [x**k for x in np.linspace(0, 1)],
        s=15,
        color=plt.cm.summer(k / 30),
    )
    plt.scatter(
        np.linspace(2, 1),
        [x**k for x in np.linspace(0, 1)],
        s=15,
        color=plt.cm.winter(k / 30),
    )
    plt.scatter(
        np.linspace(0, 1),
        [-(x**k) for x in np.linspace(0, 1)],
        s=15,
        color=plt.cm.autumn(k / 30),
    )
    plt.scatter(
        np.linspace(2, 1),
        [-(x**k) for x in np.linspace(0, 1)],
        s=15,
        color=plt.cm.spring(k / 30),
    )
#     plt.scatter(np.linspace(-1,1),[-x**k for x in np.linspace(-1,1)])

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/08042020.PNG', facecolor='black'
)
