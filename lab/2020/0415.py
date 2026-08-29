import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=100)
    p = plt.axis('off')
    for u in np.linspace(0, 1, 30):
        plt.plot([0, 0, 1 - u, u, 0], [u, 0, 0, u, u], lw=4, color=plt.cm.PuRd(1 - u))
        plt.plot([0, 0, u, u, 0], [u, 0, 0, u, 1 - u], lw=4, color=plt.cm.PuRd(1 - u))

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
