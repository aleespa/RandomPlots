from math import pi

import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for z in np.linspace(1, 20, 100):
    n = 60
    c, s = np.cos(np.linspace(0, 2 * pi, n)), np.sin(np.linspace(0, 2 * pi, n))
    r0 = np.random.uniform(1 * z, 1.2 * z)
    r = [r0] + list(np.random.uniform(1 * z, 1.2 * z, n - 2)) + [r0]

    plt.plot(
        [c[i] * r[i] for i in range(n)],
        [s[i] * r[i] for i in range(n)],
        lw=1.1,
        color=plt.cm.cool(z / 20),
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/20032020.PNG', facecolor='black')
