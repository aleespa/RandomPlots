import matplotlib.pylab as plt
import numpy as np

from tools.settings import Settings

colors = ['#3a3663', '#414977', '#476589', '#4c7c9a', '#58c0e7']


def generate(settings=Settings()):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=150)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    for _ in range(100):
        R = np.random.uniform(-5, 5, 4)
        ax.fill_between(
            [0, -R[0], -R[1], 0],
            [0, -R[2], R[3], 0],
            alpha=0.25,
            color=np.random.choice(colors),
        )
        ax.plot([0, -R[0], -R[1], 0], [0, -R[2], R[3], 0], alpha=0.2, color='white', lw=1)
    plt.savefig(
        f'outputs/{settings.filename}/figure_1.png', facecolor='black'
    )

    fig, _ = plt.subplots(figsize=(12, 12), dpi=150)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    for _ in range(100):
        R = np.random.uniform(-5, 5, 4)
        ax.fill_between(
            [0, -R[0], -R[0], 0],
            [0, -R[1], R[1], 0],
            alpha=0.25,
            color=np.random.choice(colors),
        )
        ax.plot([0, -R[0], -R[0], 0], [0, -R[1], R[1], 0], alpha=0.2, color='white', lw=1)
    plt.savefig(
        f'outputs/{settings.filename}/figure_2.png', facecolor='black'
    )
