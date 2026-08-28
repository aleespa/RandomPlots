import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def web(X, Y, n, colors):
    plt.figure(figsize=(14, 14), facecolor='black')
    plt.axis('off')
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.scatter(X, Y, color='white', zorder=n**2, alpha=0.6)
    for i in range(n):
        for j in range(n):
            plt.plot(
                [list(zip(X, Y))[i][0], list(zip(X, Y))[j][0]],
                [list(zip(X, Y))[i][1], list(zip(X, Y))[j][1]],
                color=colors[j * i],
                lw=3,
                alpha=0.8,
            )


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 15
    rng = settings.rng
    X, Y = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    colors = [plt.cm.cool(rng.uniform(0, 1)) for _ in range(n**2)]

    web(X, Y, n, colors)
    settings.save_frame('black')
    for i in range(1, 420):
        X = X + rng.normal(0, 0.5, n)
        Y = Y + rng.normal(0, 0.5, n)
        web(X, Y, n, colors)
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
