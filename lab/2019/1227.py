import gc

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=1000)
    plt.axis('off')
    for z in range(30):
        xx, yy = np.meshgrid(np.linspace(0, 1, z), np.linspace(0, 1, z))
        plt.plot(xx, yy, marker='.', linestyle='none')
    settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
