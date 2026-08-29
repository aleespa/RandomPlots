import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for k in range(2, 30):
        plt.scatter(
            np.linspace(0, 1),
            [x**k for x in np.linspace(0, 1)],
            s=15,
            color=plt.cm.summer(k / 30),
        )
        plt.scatter(
            np.linspace(2, 1),
            [x**k for x in np.linspace(0, 1)],
            s=15,
            color=plt.cm.winter(k / 30),
        )
        plt.scatter(
            np.linspace(0, 1),
            [-(x**k) for x in np.linspace(0, 1)],
            s=15,
            color=plt.cm.autumn(k / 30),
        )
        plt.scatter(
            np.linspace(2, 1),
            [-(x**k) for x in np.linspace(0, 1)],
            s=15,
            color=plt.cm.spring(k / 30),
        )
    #     plt.scatter(np.linspace(-1,1),[-x**k for x in np.linspace(-1,1)])

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
