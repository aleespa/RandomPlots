import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(15, 15), facecolor='black', dpi=400)
    p = plt.axis('off')
    n = 130
    X, Y = [x for x in np.linspace(-1, 1, n)], [x**2 for x in np.linspace(-1, 1, n)]
    for i in range(n):
        for j in range(n):
            if (i + 1) % (j + 1) == 1:
                plt.plot(
                    [X[i], X[j]],
                    [Y[i], Y[j]],
                    lw=2,
                    alpha=0.8,
                    color=plt.cm.viridis(i * j / (n**2) + 0.25),
                )

    settings.save_frame('black')

if __name__ == '__main__':
    generate()
