import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')

for u in np.linspace(0.02, 0.1, 55):
    K = np.random.uniform(1, 15)
    Sn = np.append([1], abs(np.random.normal(1 + u, 0.01, 60)))
    plt.plot(
        K * Sn.cumprod() / (K + (Sn.cumprod() - 1)),
        color=plt.cm.rainbow(np.random.uniform(0, 1)),
        lw=np.random.uniform(2, 4.5),
        alpha=0.85,
    )

plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/12032020.png', facecolor='black')
