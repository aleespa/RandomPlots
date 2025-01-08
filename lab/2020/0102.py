import matplotlib.pylab as plt
import numpy as np

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-1.2, 1.2)
p = plt.ylim(-1.2, 1.2)
colors = ['#cc0000', '#cc0033', '#999999', '#660000'] * 500
for p in list(range(1, 50)) + [500]:
    plt.plot(
        np.linspace(0, 1, 800),
        [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
        lw=3,
        color=colors[p],
    )
    plt.plot(
        np.linspace(0, 1, 800),
        [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
        lw=3,
        color=colors[p],
    )
    plt.plot(
        [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
        np.linspace(0, 1, 800),
        lw=3,
        color=colors[p],
    )
    plt.plot(
        [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
        np.linspace(0, -1, 800),
        lw=3,
        color=colors[p],
    )
plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/02012020/02012020_{1}.png',
    facecolor='black',
)

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-1.2, 1.2)
p = plt.ylim(-1.2, 1.2)
colors = ['#cc0000', '#cc0033', '#999999', '#660000'] * 500
for p in list(range(1, 50)) + [500]:
    if p == 1:
        plt.plot(
            np.linspace(0, 1, 800),
            [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            np.linspace(0, 1, 800),
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, 1, 800),
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, -1, 800),
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
    else:
        plt.plot(
            np.linspace(0, 1, 800),
            [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
            lw=3,
            color='#999999',
        )
        plt.plot(
            np.linspace(0, 1, 800),
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            lw=3,
            color='#999999',
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, 1, 800),
            lw=3,
            color='#999999',
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, -1, 800),
            lw=3,
            color='#999999',
        )
plt.text(-0.2, 0, "${||x||}_{1}=1$", fontsize=20, color='white')
plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/02012020/02012020_{2}.png',
    facecolor='black',
)

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-1.2, 1.2)
p = plt.ylim(-1.2, 1.2)
colors = ['#cc0000', '#cc0033', '#999999', '#660000'] * 500
for p in list(range(1, 50)) + [500]:
    if p == 2:
        plt.plot(
            np.linspace(0, 1, 800),
            [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            np.linspace(0, 1, 800),
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, 1, 800),
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, -1, 800),
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
    else:
        plt.plot(
            np.linspace(0, 1, 800),
            [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
            lw=3,
            color='#999999',
        )
        plt.plot(
            np.linspace(0, 1, 800),
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            lw=3,
            color='#999999',
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, 1, 800),
            lw=3,
            color='#999999',
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, -1, 800),
            lw=3,
            color='#999999',
        )
plt.text(-0.2, 0, "${||x||}_{2}=1$", fontsize=20, color='white')
plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/02012020/02012020_{3}.png',
    facecolor='black',
)

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(-1.2, 1.2)
p = plt.ylim(-1.2, 1.2)
colors = ['#cc0000', '#cc0033', '#999999', '#660000'] * 500
for p in list(range(1, 50)) + [500]:
    if p == 500:
        plt.plot(
            np.linspace(0, 1, 800),
            [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            np.linspace(0, 1, 800),
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, 1, 800),
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, -1, 800),
            lw=4,
            color='#13a8f0',
            zorder=200,
        )
    else:
        plt.plot(
            np.linspace(0, 1, 800),
            [(1 - x**p) ** (1 / p) for x in np.linspace(0, 1, 800)],
            lw=3,
            color='#999999',
        )
        plt.plot(
            np.linspace(0, 1, 800),
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            lw=3,
            color='#999999',
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, 1, 800),
            lw=3,
            color='#999999',
        )
        plt.plot(
            [-((1 - x**p) ** (1 / p)) for x in np.linspace(0, 1, 800)],
            np.linspace(0, -1, 800),
            lw=3,
            color='#999999',
        )
plt.text(-0.2, 0, "${||x||}_{\infty}=1$", fontsize=20, color='white')
plt.savefig(
    f'C:/Users/Alejandro/Pictures/RandomPlots/02012020/02012020_{4}.png',
    facecolor='black',
)
