import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    m = 4000
    n = 4000

    s = 2200
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))
    z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))

    c = -0.4 + 0.6j
    index = np.full((n, m), True, dtype=bool)
    number_iterations = np.zeros((n, m))

    logger.info(f"Starting calculation")
    for i in range(256):
        z[index] = z[index] ** 2 + c
        index[np.abs(z) > 2] = False
        number_iterations[index] = i

    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.flipud(number_iterations), cmap='gist_earth')

    settings.save_to_png(fig, 'k')
    logger.info(f"Finished")


if __name__ == '__main__':
    generate()
