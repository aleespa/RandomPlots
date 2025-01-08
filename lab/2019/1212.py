import matplotlib.pylab as plt
import numpy as np

n = 15
X, Y = np.random.uniform(-1, 1, n), np.random.uniform(-1, 1, n)
colors = [plt.cm.cool(np.random.uniform(0, 1)) for x in range(n**2)]


def web(X, Y, n):
    p = plt.figure(figsize=(14, 14), facecolor='black')
    p = plt.axis('off')
    p = plt.xlim(-20, 20)
    p = plt.ylim(-20, 20)
    p = plt.scatter(X, Y, color='white', zorder=n**2, alpha=0.6)
    for i in range(n):
        for j in range(n):
            p = plt.plot(
                [list(zip(X, Y))[i][0], list(zip(X, Y))[j][0]],
                [list(zip(X, Y))[i][1], list(zip(X, Y))[j][1]],
                color=colors[j * i],
                lw=3,
                alpha=0.8,
            )


web(X, Y, n)
plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/12122019/plot{0}.png', facecolor='black'
)
for i in range(1, 420):
    X = X + np.random.normal(0, 0.5, n)
    Y = Y + np.random.normal(0, 0.5, n)
    web(X, Y, n)
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/12122019/plot{i}.png',
        facecolor='black',
    )
    plt.clf()
