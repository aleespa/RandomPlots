import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    p = plt.xlim(-1, -0.5)
    p = plt.ylim(-1, -0.5)
    for _ in range(14):
        X, Y = [settings.rng.uniform(-1, 1) for y in np.linspace(0, 1, 3000)], [
            settings.rng.uniform(-1, 1) for y in np.linspace(0, 1, 3000)
        ]
        # plt.fill_between(X,Y, color='black', alpha=0.4)
        plt.plot(X, Y, alpha=0.7, lw=0.4)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
