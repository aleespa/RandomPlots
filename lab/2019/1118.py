import gc

import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def Random(n, rng):
    x = [0]
    y = [0]
    for i in range(n):
        L = [1, 0, -1]
        s = rng.choice(L)
        x.append(x[-1] + s)
        if s == 0:
            y.append(y[-1] + rng.choice([-1, 1]))
        else:
            y.append(y[-1])
    return x, y


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
    plt.axis('off')
    plt.xlim(-20, 20.1)
    plt.ylim(-20, 20.1)
    plt.scatter([0], [0], color='white', zorder=400, s=45)
    colors = (
        ['#5f865a', '#65a659', '#77bb5d', '#8ed067', '#abdf7b'] * 3
        + ['#ff5e5e', '#ec7f7f', '#ee96be', '#f49ade', '#feb4ff'] * 3
        + ['#3a3663', '#414977', '#476589', '#4c7c9a', '#50919b'] * 3
    )
    for num in range(45):
        X, Y = Random(400, settings.rng)
        for i in range(1, 9):
            plt.plot(
                X[: i * 50], Y[: i * 50], color=colors[num], zorder=3, lw=4, alpha=0.8
            )
            settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
