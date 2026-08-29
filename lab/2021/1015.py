import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    route = '/mnt/c/Users/Alejandro López/Pictures/RandomPlots/'

    r1 = 1j
    r2 = -1j
    r3 = -1
    P = np.vectorize(lambda x: (x - r1) * (x - r2) * (x - r3))
    Pp = np.vectorize(
        lambda x: r2 * (r3 - 2 * x) + r1 * (r2 + r3 - 2 * x) + x * (-2 * r3 + 3 * x)
    )

    X = np.linspace(-1.25, 0.4, 50)
    Y = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(X, Y)
    Z0 = xx + yy * 1j

    plt.figure(figsize=(14, 14), facecolor='k', dpi=200)
    plt.axis('off')
    plt.scatter(Z0.real, Z0.imag, s=30, c=np.abs(P(Z0)) ** 0.7, cmap=plt.cm.jet)
    plt.xlim(-1.25, 0.4)
    plt.ylim(-1.5, 1.5)
    settings.save_frame('black')

    Z = Z0

    for j in range(100):
        Z = Z - (0.02 * j**0.2) * (P(Z) / Pp(Z))
        plt.figure(figsize=(14, 14), facecolor='k', dpi=200)
        plt.axis('off')
        plt.scatter(Z.real, Z.imag, s=30, c=np.abs(P(Z0)) ** 0.7, cmap=plt.cm.jet)
        plt.xlim(-1.25, 0.4)
        plt.ylim(-1.5, 1.5)
        settings.save_frame('black')
        plt.close()

if __name__ == '__main__':
    generate()
