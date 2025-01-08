import matplotlib.pylab as plt
import numpy as np

path = '/mnt/c/Users/Alejandro Lopez/Pictures/RandomPlots/'

t = np.linspace(0, 8 * np.pi, 2000)
i = 2
N = 16
tail = 220

d = np.arange(2, N + 2)
s = np.random.uniform(-0.8, 0.8, N)
s[8] = 1.1
c = ['#075791'] + ['w'] * (N - 1)

for f, i in enumerate(range(1, 2000, 5)):
    i0 = max(i - tail, 0)
    plt.figure(figsize=(12, 12), facecolor='black', dpi=100)
    plt.axis('off')

    plt.scatter(0, 0, color='w', s=100, zorder=2 * N)
    plt.scatter(0, 0, color='w', s=150, zorder=2 * N, alpha=0.5)
    plt.scatter(0, 0, color='w', s=250, zorder=2 * N, alpha=0.4)
    plt.scatter(0, 0, color='w', s=300, zorder=2 * N, alpha=0.3)
    plt.scatter(0, 0, color='w', s=500, zorder=2 * N, alpha=0.2)
    plt.scatter(0, 0, color='w', s=700, zorder=2 * N, alpha=0.1)

    for j in range(1, len(d)):
        x = d[j] * np.cos(s[j] * t[i0:i])
        y = d[j] * np.sin(s[j] * t[i0:i])
        for m in range(len(x)):
            plt.plot(
                x[m : m + 2],
                y[m : m + 2],
                zorder=j,
                color=plt.cm.cubehelix(m / len(x)),
                lw=3,
            )

        plt.scatter(x[-1], y[-1], s=100, color=c[j], zorder=j + 1)

    plt.xlim(-N - 2.5, N + 2.5)
    plt.ylim(-N - 2.5, N + 2.5)
    plt.savefig(path + f'15012022/v1/plot_{f}.png', facecolor='black')
    plt.close()

for f, i in enumerate(range(1, 2000, 5)):
    i0 = max(i - tail, 0)
    plt.figure(figsize=(12, 12), facecolor='black', dpi=100)
    plt.axis('off')

    plt.scatter(0, 0, color='w', s=100, zorder=2 * N)
    plt.scatter(0, 0, color='w', s=150, zorder=2 * N, alpha=0.5)
    plt.scatter(0, 0, color='w', s=250, zorder=2 * N, alpha=0.4)
    plt.scatter(0, 0, color='w', s=300, zorder=2 * N, alpha=0.3)
    plt.scatter(0, 0, color='w', s=500, zorder=2 * N, alpha=0.2)
    plt.scatter(0, 0, color='w', s=700, zorder=2 * N, alpha=0.1)

    j = 8
    for k in range(N):
        if k == j:
            pass
        else:

            x0 = d[j] * np.cos(s[j] * t[i0:i])
            y0 = d[j] * np.sin(s[j] * t[i0:i])

            xk = d[k] * np.cos(s[k] * t[i0:i])
            yk = d[k] * np.sin(s[k] * t[i0:i])

            r = np.linalg.norm(np.array([xk, yk]) - np.array([x0, y0]), axis=0)
            a = np.arctan2(yk - y0, xk - x0)

            x = r * np.cos(a)
            y = r * np.sin(a)
            for m in range(len(x)):
                plt.plot(
                    x[m : m + 2],
                    y[m : m + 2],
                    zorder=j,
                    color=plt.cm.gnuplot(m / len(x)),
                    lw=3,
                )

            plt.scatter(x[-1], y[-1], s=100, color=c[j], zorder=j + 1)
    plt.xlim(-N - j - 5, N + j + 5)
    plt.ylim(-N - j - 5, N + j + 5)
    plt.savefig(path + f'15012022/v2/plot_{f}.png', facecolor='black')
    plt.close()
