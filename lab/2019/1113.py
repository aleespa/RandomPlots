import matplotlib.pylab as plt
import numpy as np

colors = ['#3a3663', '#414977', '#476589', '#4c7c9a', '#58c0e7']
p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for _ in range(100):
    R = np.random.uniform(-5, 5, 4)
    plt.fill_between(
        [0, -R[0], -R[1], 0],
        [0, -R[2], R[3], 0],
        alpha=0.25,
        color=np.random.choice(colors),
    )
    plt.plot([0, -R[0], -R[1], 0], [0, -R[2], R[3], 0], alpha=0.2, color='white', lw=1)
p = plt.savefig(
    'C:/Users/Alejandro/Pictures/RandomPlots/13112019_1.PNG', facecolor='black'
)

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for _ in range(100):
    R = np.random.uniform(-5, 5, 4)
    plt.fill_between(
        [0, -R[0], -R[0], 0],
        [0, -R[1], R[1], 0],
        alpha=0.25,
        color=np.random.choice(colors),
    )
    plt.plot([0, -R[0], -R[0], 0], [0, -R[1], R[1], 0], alpha=0.2, color='white', lw=1)
p = plt.savefig(
    'C:/Users/Alejandro/Pictures/RandomPlots/13112019_2.PNG', facecolor='black'
)
