import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 800
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(-10.1, 10.1)
    plt.ylim(-10.1, 10.1)
    X, Y = settings.rng.uniform(-10, 10, n), settings.rng.uniform(-10, 10, n)
    plt.scatter(X, Y, color='white', zorder=n**2, alpha=0.7, s=8)
    for i in range(n):
        plt.plot(
            [X[i], X[i] * 0.7],
            [Y[i], Y[i] * 1.5],
            alpha=0.8,
            lw=settings.rng.uniform(1, 5),
        )
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
