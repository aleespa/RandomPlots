import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for y in range(4, 80):
        Z = rng.uniform(0, 1, 6)
        plt.fill_between([Z[0], Z[1]], [Z[2], Z[3]], [Z[4], Z[5]], alpha=0.5)
    settings.save_to_png(fig, 'black')

    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for y in range(4, 100):
        Z = rng.normal(0, 1, 6)
        plt.fill_between([Z[0], Z[1]], [Z[2], Z[3]], [Z[4], Z[5]], alpha=0.5)
    settings.save_to_png(fig, 'black')

    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for y in range(4, 100):
        Z = rng.gamma(1, 3, 6)
        plt.fill_between([Z[0], Z[1]], [Z[2], Z[3]], [Z[4], Z[5]], alpha=0.5)
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
