import gc

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = ['#79c562', '#4ea150', '#edf69f', '#ecee43']
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for x in np.linspace(-1, 1, 30):
        for y in np.linspace(-1, 1, 30):
            plt.plot(
                [x, y + x], [x, y - x], alpha=0.75, lw=1, color=settings.rng.choice(colors)
            )
    settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    generate()
