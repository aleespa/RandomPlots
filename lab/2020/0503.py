import matplotlib.pylab as plt
import numpy as np

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = X**2 + Y**2
V = X**2 - Y**2
speed = np.sqrt(U**2 + V**2)
lw = 5 * speed / speed.max()
p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
plt.streamplot(
    X,
    Y,
    U,
    V,
    density=[2, 2],
    arrowsize=1,
    cmap='summer',
    color=U,
    linewidth=lw**0.85,
    arrowstyle='fancy',
)

p = plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/03052020.PNG', facecolor='black'
)
