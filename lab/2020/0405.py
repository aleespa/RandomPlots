import matplotlib.pylab as plt
import numpy as np


def F_Cantor(n):
    if n == 0:
        return lambda x: x
    else:
        return (
            lambda x: 0.5 * F_Cantor(n - 1)(3 * x) * (0 <= x) * (x <= 1 / 3)
            + 0.5 * (1 / 3 < x) * (x <= 2 / 3)
            + (0.5 + 0.5 * F_Cantor(n - 1)(3 * x - 2)) * (2 / 3 < x) * (x <= 1)
        )


p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for n in reversed(range(10)):
    plt.fill_between(
        np.linspace(0, 1, 100),
        [F_Cantor(n)(x) for x in np.linspace(0, 1, 100)],
        alpha=0.4,
        color=plt.cm.RdPu(n / 10),
    )
    plt.fill_between(
        np.linspace(1, 2, 100),
        [1 - F_Cantor(n)(x) for x in np.linspace(0, 1, 100)],
        alpha=0.4,
        color=plt.cm.RdPu(1 - n / 10),
    )
    plt.fill_between(
        np.linspace(0, 1, 100),
        [-F_Cantor(n)(x) for x in np.linspace(0, 1, 100)],
        alpha=0.4,
        color=plt.cm.RdPu(1 - n / 10),
    )
    plt.fill_between(
        np.linspace(1, 2, 100),
        [F_Cantor(n)(x) - 1 for x in np.linspace(0, 1, 100)],
        alpha=0.4,
        color=plt.cm.RdPu(n / 10),
    )

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/05042020.PNG', facecolor='black'
)
