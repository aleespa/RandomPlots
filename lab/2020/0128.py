import matplotlib.pylab as plt

p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
p = plt.axis('off')
for z in range(-25, 25):
    plt.plot(
        [0, 1],
        [0, z],
        lw=0.35 * (abs(z) + 3) ** 0.5,
        alpha=1,
        zorder=200 - z,
        color=plt.cm.rainbow(((z + 25) / 50)),
    )
    plt.plot(
        [2, 1],
        [0, z],
        lw=0.35 * (abs(z) + 3) ** 0.5,
        alpha=1,
        zorder=200 - z,
        color=plt.cm.rainbow(1 - ((z + 25) / 50)),
    )
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/28012020.png', facecolor='black')
