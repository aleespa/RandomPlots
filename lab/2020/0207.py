import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.ylim(-150, 150)
random_lw = np.random.uniform(0.6, 5, 40)
for i, z in zip(range(40), np.linspace(1, 40, 40)):
    plt.plot(
        np.linspace(0, 0.5),
        [1 / (x) * z for x in np.linspace(0.01, 0.5)],
        lw=random_lw[i],
        color=plt.cm.viridis(np.random.uniform(0, 1)),
    )
    plt.plot(
        np.linspace(0, 0.5),
        [-1 / (x) * z for x in np.linspace(0.01, 0.5)],
        lw=random_lw[i],
        color=plt.cm.plasma(np.random.uniform(0, 1)),
    )
    plt.plot(
        -np.linspace(0, 0.5),
        [-1 / (x) * z for x in np.linspace(0.01, 0.5)],
        lw=random_lw[i],
        color=plt.cm.viridis(np.random.uniform(0, 1)),
    )
    plt.plot(
        -np.linspace(0, 0.5),
        [1 / (x) * z for x in np.linspace(0.01, 0.5)],
        lw=random_lw[i],
        color=plt.cm.plasma(np.random.uniform(0, 1)),
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/07022020.png', facecolor='black')
