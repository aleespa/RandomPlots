import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 180
    X, Y = settings.rng.uniform(-10, 10, n), settings.rng.uniform(-10, 10, n)

    for y in np.linspace(1, 0, 250):
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.axis('off')
        plt.xlim(-10.1, 10.1)
        plt.ylim(-10.1, 10.1)
        plt.scatter(X, Y, zorder=n**2, alpha=1, s=9, color='#ff7142')
        for i in range(n):
            plt.plot([X[i], X[i] * y], [Y[i], Y[i]], alpha=0.8, color='white', lw=2)
        settings.save_frame('black')
    for y in np.linspace(1, 0, 250):
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.axis('off')
        plt.xlim(-10.1, 10.1)
        plt.ylim(-10.1, 10.1)
        plt.scatter(X, Y, zorder=n**2, alpha=1, s=9, color='#ff7142')
        for i in range(n):
            plt.plot([X[i], 0], [Y[i], Y[i] * y], alpha=0.8, color='white', lw=2)
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
