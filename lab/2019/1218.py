import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 2000

    lws = settings.rng.uniform(1, 3, n)
    X, Y = settings.rng.uniform(-10, 10, n), settings.rng.uniform(-10, 10, n)
    m = 0
    for y in range(30):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(-10.1, 10.1)
        plt.ylim(-10.1, 10.1)
        plt.scatter(X, Y, color='white', zorder=n**2, alpha=0.9, s=lws * 1.2)
        settings.save_frame('black')
        m += 1
    for y in np.linspace(1, 0.65, 18):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(-10.1, 10.1)
        plt.ylim(-10.1, 10.1)
        plt.scatter(X, Y, color='white', zorder=n**2, alpha=0.8, s=lws)
        for i in range(n):
            plt.plot(
                [X[i], X[i] * y],
                [Y[i], Y[i] * y],
                alpha=1,
                color='#4584ff',
                lw=lws[i] * 1.2,
            )
            plt.plot(
                [X[i], X[i] * y],
                [Y[i], Y[i] * y],
                alpha=0.8,
                color='white',
                lw=lws[i] / 1.5,
            )
        settings.save_frame('black')
        m += 1
    for y in np.linspace(1, 3, 18):
        plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
        plt.axis('off')
        plt.xlim(-10.1, 10.1)
        plt.ylim(-10.1, 10.1)
        plt.scatter(X * y, Y * y, color='white', zorder=n**2, alpha=0.8, s=lws)
        for i in range(n):
            plt.plot(
                [X[i] * y, X[i] * 0.65 * y],
                [Y[i] * y, Y[i] * 0.65 * (y**0.85)],
                alpha=1,
                color='#4584ff',
                lw=lws[i] * 1.2,
            )
            plt.plot(
                [X[i] * y, X[i] * 0.65 * y],
                [Y[i] * y, Y[i] * 0.65 * (y**0.85)],
                alpha=0.8,
                color='white',
                lw=lws[i] / 1.5,
            )
        settings.save_frame('black')
        m += 1


if __name__ == '__main__':
    generate()
